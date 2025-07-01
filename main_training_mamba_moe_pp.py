import math
import os
from pathlib import Path

import fire
import torch
import torch.nn.functional as F
import torch.optim as optim
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.moe_utils import (
    act_ckpt_moe,
    fully_shard_moe,
    get_total_exp_and_active_params,
    init_moe,
    set_pp_layers,
)
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.distributed.tensor import DTensor
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import CheckpointerFSDP2
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils_moe_pp import (
    get_profiler,
    setup,
    setup_environ_flags,
    train_moe_pp,
)

"""
MoE + EP training. Requires torch nightly > 2.6. For use with the branch here: https://github.com/garrett361/mamba/pull/4
"""


@record
def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    torch.cuda.set_device(local_rank)
    setup(cfg)
    torch.cuda.empty_cache()
    setup_environ_flags()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        Path.home(), ".triton", "cache", str(local_rank)
    )

    # NOTE: @goon - Seems to help with NCCL stability to start with a barrier.
    dist.barrier()

    # get model
    mamba_config = get_model_config(cfg.model_variant)
    mamba_config.moe_cfg["moe_impl"] = cfg.moe_impl
    # HACK: @goon - force bias = True when using loss_free_balancing_lr
    if cfg.loss_free_balancing_lr:
        if cfg.loss_free_balancing_lr < 0:
            raise ValueError(f"{cfg.loss_free_balancing_lr=} must be non-negative.")
        mamba_config.moe_cfg["gate_bias"] = True

    # NOTE: @goon - set return_logits = True to avoid returning a CausalLMOutput object and just
    # return logit torch.Tensor. A feature of the mamba_moe branch of mamba
    mamba_config.return_logits = True

    if not rank:
        print(f"{mamba_config.moe_cfg=}")

    # Check and update cfg vocab size, if needed.
    cfg_vocab = cfg.vocab_size
    model_vocab = mamba_config.vocab_size
    if cfg_vocab != model_vocab:
        if not rank:
            print(
                f"Config vocab size ({cfg_vocab}) does not match model vocab size ({model_vocab})."
                " Adjusting config vocab size to match model."
            )
        cfg.vocab_size = mamba_config.vocab_size

    # Mesh setup
    assert world_size >= cfg.ep_degree, (
        f"{world_size=} must be at least as large as {cfg.ep_degree=}"
    )
    assert world_size % cfg.ep_degree == 0, (
        f"{world_size=} must be divisible by {cfg.ep_degree=}"
    )

    pp_degree = world_size // cfg.ep_degree
    mesh = init_device_mesh(
        "cuda", (pp_degree, cfg.ep_degree), mesh_dim_names=("pp", "ep")
    )

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    # All PP members get the same data, so it's EP-way data parallel
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, mesh["ep"].get_local_rank(), cfg.ep_degree)
    else:
        train_loader = get_dummy_loader(cfg, mesh["ep"].get_local_rank(), cfg.ep_degree)
    if rank == 0:
        print("Datasets constructed!")

    if cfg.sharding_strategy == "hsdp":
        raise NotImplementedError("TODO: hsdp")
    else:
        fsdp_mesh = mesh["ep"]

    if rank == 0:
        print(f"{mesh=}")
        # Count for the full model on the meta device to avoid inaccurate counts due to EP
        with torch.device("meta"):
            total, exp, active = get_total_exp_and_active_params(
                MambaLMHeadModel(mamba_config)
            )
        print(
            f"\n--> Logical model has {total / 1e9:.2f}B params\n"
            f"\t{exp / 1e9:.2f}B Routed Expert Params\n"
            f"\t{active / 1e9:.2f}B Active Params Per Token"
        )

    # Model building order:
    # 1. Create model, maybe on meta device.
    # 2. adjust layers for pp (set_pp_layers)
    # 3. Activation checkpointing, if applicable
    # 4. fully_shard
    # 5. init weights, if meta device was used
    # 6. Build optimizers
    # 7. Optionally compile
    # 8. Build pipeline stages
    # 10. Build pipeline schedule

    if cfg.low_cpu_fsdp:
        if rank == 0:
            print("Building model on meta device...")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config, ep_mesh=mesh["ep"])
    else:
        model = MambaLMHeadModel(mamba_config, ep_mesh=mesh["ep"])

    if rank == 0:
        print(f"\nFull model: {model}")

    set_pp_layers(
        model,
        n_stages=pp_degree,
        stage_idx=mesh["pp"].get_local_rank(),
    )

    if mesh["ep"].get_local_rank() == 0:
        print(f"\nPP model on {mesh['pp'].get_local_rank()=}: {model}")

    # NOTE: @goon - Sanity checking param count:
    if rank == 0:
        # NOTE: @goon - DTensor.numel() will report the full logical parameter counts, whereas we
        # want the actual count of local params here.
        total_params_local = sum(
            p.to_local().numel() if isinstance(p, DTensor) else p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        print(f"\n--> Local model has {total_params_local / 1e9} Billion params\n")

    # AC
    if cfg.fsdp_activation_checkpointing:
        act_ckpt_moe(model, cfg.act_ckpt_mixer_only)

    dtype = torch.bfloat16
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
    if not rank:
        print("Running fully_shard ...")
    dist.barrier()
    fully_shard_moe(
        model,
        fsdp_mesh=mesh["ep"],
        ep_fsdp_mesh=None,
        mp_policy=mp_policy,
        reshard_lm_head_after_fwd=cfg.reshard_lm_head_after_fwd,
        explicit_fwd_prefetch=cfg.explicit_fwd_prefetch,
        explicit_bwd_prefetch=cfg.explicit_bwd_prefetch,
    )

    if cfg.low_cpu_fsdp:
        if rank == 0:
            print("Moving meta model to CUDA...")
        init_moe(model)

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print("--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        foreach=cfg.foreach,
        fused=cfg.fused,
    )

    # PP setup

    # PP Metadata
    is_first = mesh["pp"].get_local_rank() == 0
    is_last = mesh["pp"].get_local_rank() == mesh["pp"].size() - 1

    # Set input/output tensor shapes to avoid PP from trying (and often failing) to auto-determine
    # shapes.

    assert cfg.n_microbatches, f"{cfg.n_microbatches=}"
    assert cfg.batch_size % cfg.n_microbatches == 0, (
        f"{cfg.n_microbatches=}, {cfg.batch_size=}"
    )

    if is_first:
        input_args = torch.empty(
            cfg.batch_size // cfg.n_microbatches,
            cfg.seq_length,
            device="meta",
            dtype=torch.int32,
        )
    else:
        input_args = torch.empty(
            cfg.batch_size // cfg.n_microbatches,
            cfg.seq_length,
            model.config.d_model,
            dtype=dtype,
            device="meta",
        )
    if is_last:
        output_args = torch.empty(
            cfg.batch_size // cfg.n_microbatches,
            cfg.seq_length,
            cfg.vocab_size,
            dtype=dtype,
            device="meta",
        )
    else:
        output_args = torch.randn(
            cfg.batch_size // cfg.n_microbatches,
            cfg.seq_length,
            model.config.d_model,
            dtype=dtype,
            device="meta",
        )

    if not rank:
        print("Creating PipelineStage ...")
    dist.barrier()
    stage = PipelineStage(
        model,
        mesh["pp"].get_local_rank(),
        mesh["pp"].size(),
        local_rank,  # Is an int fine here?
        group=mesh["pp"].get_group(),
        input_args=input_args,
        output_args=output_args,
    )
    # NOTE: @goon -  PipleineStage doesn't have a nice repr; no reason to print.

    def flattened_cross_entropy(
        input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(input.view(-1, input.size(-1)), target.view(-1).long())

    if not rank:
        print("Creating PP Schedule ...")
    dist.barrier()
    pp_schedule = Schedule1F1B(
        stage, cfg.n_microbatches, loss_fn=flattened_cross_entropy
    )
    # NOTE: @goon -  Schedules don't have a nice repr; no reason to print.

    # optionally load from checkpoint (when continue pretraining)
    if cfg.skip_ckpt:
        checkpointer = None
        tokens_seen = start_step = 0
        is_resuming = False
    else:
        checkpointer = CheckpointerFSDP2(
            cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
        )
        model, optimizer, _, start_step, tokens_seen, is_resuming = checkpointer.load(
            model,
            optimizer,
            None,
            path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
            if not os.path.isfile(cfg.ckpt_load_path)
            else cfg.ckpt_load_path,
            strict=False,
        )

        if not is_resuming:
            start_step = 0
            # Override loaded optim hyperparams with the current values
            for g in optimizer.param_groups:
                g["initial_lr"] = cfg.learning_rate

    # LR schedule
    # linear decay for annealing
    if cfg.training_stage == "annealing":
        warmup_interval = 1000
        schedule = (
            lambda x: x / warmup_interval
            if x < warmup_interval
            else 1 - (x - warmup_interval) / (cfg.num_steps - warmup_interval)
        )
    elif cfg.training_stage == "cosine":
        # cosine decay
        warmup_interval = min(2000, cfg.num_steps // 20)
        schedule = lambda x: min(
            1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
            0.1
            + 0.5
            * (1 - 0.1)
            * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )
    elif cfg.training_stage == "constant":
        warmup_interval = 2000
        schedule = lambda x: (min(x, warmup_interval) / warmup_interval)
    else:
        schedule = lambda x: 1.0 + (0.75 - 1.0) * (x / 32000) if x <= 32000 else 0.75

    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    train_moe_pp(
        cfg,
        model,
        pp_schedule,
        mesh,
        is_first,
        is_last,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        profiler,
        checkpointer,
        start_step,
        tokens_seen,
    )

    if not cfg.skip_ckpt:
        checkpointer.save_single_file(cfg.num_steps, model)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
