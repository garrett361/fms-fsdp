import math
import os
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.moe import MoE
from torch import distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils import Checkpointer
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
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        Path.home(), ".triton", "cache", str(local_rank)
    )

    # get model
    config_data = get_model_config(cfg.model_variant)
    mamba_config = MambaConfig(**config_data)
    if cfg.force_equal_loads:
        mamba_config.moe_cfg["_force_equal_loads"] = True
    if not rank:
        print(f"{mamba_config.moe_cfg=}")

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
    assert world_size % cfg.ep_degree == 0, (
        f"{world_size=} must be divisible by {cfg.ep_degree=}"
    )

    # Cases:
    # 1. ep_degree = 1: full replication, no ep_mesh
    # 2. ep_degree = world_size: ep_mesh is the world
    # 3. world_size > ep_degree > world_size: 2D mesh, experts distributed along slice .
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
            total_params = sum(
                p.numel()
                for p in MambaLMHeadModel(mamba_config).parameters()
                if p.requires_grad
            )
        print(f"\n--> Logical model has {total_params / 1e6} Million params\n")
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
        print(f"\n--> Local model has {total_params_local / 1e6} Million params\n")

    # AC
    if cfg.fsdp_activation_checkpointing:
        for layer_index, block in model.backbone.layers.items():
            model.backbone.layers[layer_index] = checkpoint_wrapper(
                block, preserve_rng_state=False
            )

    # TODO: @goon - selective AC

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
    # Assumption: no tied params
    fully_shard(model.lm_head, mesh=fsdp_mesh, mp_policy=mp_policy)
    fully_shard(model.backbone.embedding, mesh=fsdp_mesh, mp_policy=mp_policy)
    # NOTE: @goon - model.backbone.layers is a module_dict on the MoE branch
    for idx, block in model.backbone.layers.items():
        # Cases:
        # 1. ep_degree = 1: full replication, fully shard with the fsdp_mesh
        # 2. ep_degree = world_size: no expert replication at all. Ignore experts in fully_shard
        # 3. world_size > ep_degree > world_size: world_size // ep_degree expert replicas. Need
        #    to individually wrap experts using the ep_mesh because ModuleDict doesn't have a
        #    forward method.

        # The ignored_params arg requires torch nightly (> 2.6.0)
        ignored_params = set()
        if isinstance(block.mlp, MoE):
            if cfg.ep_degree == 1:
                pass
            elif cfg.ep_degree == world_size:
                # No replication in this case.
                ignored_params.add(block.mlp.experts.parameters())
            else:
                for expert in block.mlp.experts.values():
                    # Don't reshard due to comms costs
                    fully_shard(
                        expert,
                        mesh=ep_mesh["outer"],
                        mp_policy=mp_policy,
                        reshard_after_forward=False,
                    )
        is_not_last_block = int(idx) < len(model.backbone.layers) - 1
        fully_shard(
            block,
            mesh=fsdp_mesh,
            ignored_params=ignored_params,
            mp_policy=mp_policy,
            reshard_after_forward=is_not_last_block,
        )
    fully_shard(model, mesh=fsdp_mesh, reshard_after_forward=False, mp_policy=mp_policy)

    if cfg.low_cpu_fsdp:
        if rank == 0:
            print("Moving model to CUDA...")
        # Move to cuda and initialize.
        model.to_empty(device=torch.cuda.current_device())
        # NOTE: @goon - explicitly put the entire model in bfloat16. Not clear whether the ignored
        # EP experts were using bfloat16 or float32 compute.
        # TODO: @goon - figure this out and remove.
        model.to(torch.bfloat16)

        # TODO: proper normalization; just normal init for now
        for p in model.parameters():
            nn.init.normal_(p)
        nn.init.normal_(model.backbone.embedding.weight, std=0.02)

    else:
        # Must also manually move the ignored experts to cuda, as fully_shard doesn't do so.
        if cfg.ep:
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
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
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
