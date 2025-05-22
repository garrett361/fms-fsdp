import math
import os
from pathlib import Path

import fire
import torch
import torch.optim as optim
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.moe import MoE
from mamba_ssm.moe_utils import (
    act_ckpt_moe,
    fully_shard_moe,
    get_total_exp_and_active_params,
    init_meta_moe,
)
from torch import distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import CheckpointerFSDP2
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)

"""
MoE + EP training. Requires torch nightly > 2.6. For use with the branch here: https://github.com/garrett361/mamba/pull/4
"""


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

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

    # FSDP
    if cfg.sharding_strategy == "hsdp":
        fsdp_mesh = init_device_mesh(
            "cuda",
            (world_size // torch.cuda.device_count(), torch.cuda.device_count()),
            mesh_dim_names=("outer", "inner"),
        )
    else:
        fsdp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))

    # NOTE: @goon - In context parallel, training was much more stable when using separate meshes
    # for fsdp and CP. Trying the same thing here with EP, but not sure it matters. Don't think it
    # should, in principle. Also, we may just need separate meshes in the future for more complex
    # scenarios.
    assert world_size >= cfg.ep_degree, (
        f"{world_size=} must be at least as large as {cfg.ep_degree=}"
    )
    assert world_size % cfg.ep_degree == 0, (
        f"{world_size=} must be divisible by {cfg.ep_degree=}"
    )

    # Cases:
    # 1. ep_degree = 1: full replication, no ep_mesh
    # 2. ep_degree = world_size: ep_mesh is the world
    # 3. world_size > ep_degree > world_size: 2D mesh with (DP, EP) dims, experts distributed along
    #    slice.
    if cfg.ep_degree == 1:
        ep_mesh = None
    elif cfg.ep_degree == world_size:
        ep_mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("inner",),
        )
    else:
        ep_mesh = init_device_mesh(
            "cuda",
            (world_size // cfg.ep_degree, cfg.ep_degree),
            mesh_dim_names=("outer", "inner"),
        )

    if rank == 0:
        print(f"{ep_mesh=}")
        print(f"{fsdp_mesh=}")
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
    # 2. Activation checkpointing, if applicable
    # 3. Compile, if applicable
    # 4. fully_shard
    # 5. init weights, if meta device was used
    if cfg.low_cpu_fsdp:
        if rank == 0:
            print("Building model on meta device...")
        with torch.device("meta"):
            model = MambaLMHeadModel(
                mamba_config, ep_mesh=None if ep_mesh is None else ep_mesh["inner"]
            )
    else:
        model = MambaLMHeadModel(
            mamba_config, ep_mesh=None if ep_mesh is None else ep_mesh["inner"]
        )

    # NOTE: @goon - Sanity checking param count:
    if rank == 0:
        total_params_local = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"\n--> Local model has {total_params_local / 1e9} Billion params\n")

    # AC
    if cfg.fsdp_activation_checkpointing:
        act_ckpt_moe(model, cfg.act_ckpt_mixer_only)

    # TODO: @goon - selective AC

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )

    fully_shard_moe(
        model=model,
        ep_mesh=ep_mesh,
        fsdp_mesh=fsdp_mesh,
        mp_policy=mp_policy,
        reshard_lm_head_after_fwd=cfg.reshard_lm_head_after_fwd,
        explicit_fwd_prefetch=cfg.explicit_fwd_prefetch,
        explicit_bwd_prefetch=cfg.explicit_bwd_prefetch,
        no_reshard=cfg.no_reshard,
    )

    if cfg.low_cpu_fsdp:
        if rank == 0:
            print("Moving meta model to CUDA...")
        init_meta_moe(model)

    elif cfg.ep_degree == world_size:
        # If the experts are not sharded and just ignored, then we must also manually move the
        # ignored experts to cuda, as fully_shard doesn't do so.
        for block in model.backbone.layers.values():
            if isinstance(block.mlp, MoE):
                block.mlp.experts.to(device=torch.cuda.current_device())

    if rank == 0:
        print(f"{model=}")

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
    train(
        cfg,
        model,
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
