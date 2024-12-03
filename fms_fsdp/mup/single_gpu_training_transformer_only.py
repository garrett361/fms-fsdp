import math
import os
from typing import Optional, Union
from pathlib import Path

import fire
import torch
import torch.optim as optim
from mamba_ssm.modules.block import Block
from torch import distributed as dist
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp.utils.checkpointing_utils import Checkpointer
from fms_fsdp.utils.config_utils import update_config
from fms_fsdp.utils.dataloader_utils import get_data_loader, get_dummy_loader
from fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)
from dataclasses import dataclass
from fms_fsdp.mup.transformer_only_utils import get_transformer_and_config

from fms_fsdp.mup.mup_mamba import apply_mup_init, get_mup_optim_iter


@dataclass
class mup_config:
    # model
    width: int = 512
    n_layer: int = 10
    head_dim: int = 128
    mup: bool = False
    ckpt_load_path: str = "/fsx/output/ckpt"
    ckpt_save_path: str = "/fsx/output/ckpt"

    # dataset and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/fsx/data"
    file_type: str = "arrow"
    col_name: str = "tokens"
    tokenizer_path: str = "/fsx/tokenizer"
    datasets: str = "lang=en/dataset=commoncrawl,lang=en/dataset=webhose,lang=en/dataset=github_clean,lang=de/dataset=wikipedia,lang=es/dataset=wikipedia,lang=fr/dataset=wikipedia,lang=ja/dataset=wikipedia,lang=pt/dataset=wikipedia,lang=en/dataset=wikimedia,lang=en/dataset=uspto,lang=en/dataset=pubmedcentral,lang=en/dataset=arxiv,lang=en/dataset=stackexchange"
    weights: str = "7725,500,550,28,17,22,25,8,100,500,175,250,100"
    seq_length: int = 4096
    vocab_size: int = 128256
    bos_token: Optional[int] = None
    eos_token: int = 0
    bol_token: Optional[int] = None
    eol_token: Optional[int] = None
    strip_tokens: str = ""
    logical_shards: int = 1024
    num_workers: int = 1

    # fsdp policies
    sharding_strategy: str = "hsdp"
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: Union[float, str] = 1  # percentage of blocks to apply ac
    mixed_precision: bool = True
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0
    seed: int = 2023

    # continued training spec
    resuming_dataset: bool = False

    # logging
    report_interval: int = 100
    checkpoint_interval: int = 10000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = True


def main(**kwargs):
    # get configs
    cfg = mup_config()
    update_config(cfg, **kwargs)
    print(f"{cfg=}")

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

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

    # get policy
    block = Block
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(cfg, rank, block)

    # get model
    # config_data = get_model_config(cfg.model_variant)
    # mamba_config = MambaConfig(**config_data)
    # model = MambaLMHeadModel(mamba_config)
    model, mamba_config = get_transformer_and_config(
        width=cfg.width,
        n_layer=cfg.n_layer,
        vocab_size=cfg.vocab_size,
        head_dim=cfg.head_dim,
        device="cuda",
        mup=cfg.mup,
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")
        print(f"{model=}")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

    # # FSDP
    # model = FSDP(
    #     model,
    #     auto_wrap_policy=wrapping_policy,
    #     mixed_precision=mixed_precision_policy,
    #     sharding_strategy=sharding_strategy_policy,
    #     use_orig_params=cfg.use_torch_compile,
    #     device_id=torch.cuda.current_device(),
    #     limit_all_gathers=True,
    #     param_init_fn=param_init_fn,
    # )

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print("--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=cfg.selective_checkpointing)

    # torch compile
    if cfg.use_torch_compile:
        if rank == 0:
            print("--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Model init and Optimizer
    if cfg.mup:
        apply_mup_init(model)
        optimizer = optim.AdamW(
            get_mup_optim_iter(model, lr=cfg.learning_rate, optim_type="adam"),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )
    # model, optimizer, _, start_step, tokens_seen, is_resuming = checkpointer.load(
    #     model,
    #     optimizer,
    #     None,
    #     path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
    #     if not os.path.isfile(cfg.ckpt_load_path)
    #     else cfg.ckpt_load_path,
    #     strict=False,
    # )
    start_step = tokens_seen = is_resuming = 0
    if not is_resuming:
        start_step = 0
        # Override loaded optim hyperparams with the current values
        for g in optimizer.param_groups:
            g["initial_lr"] = cfg.learning_rate

    # LR schedule
    if cfg.training_stage == "annealing":
        schedule = lambda x: 1 - x / cfg.num_steps
    else:
        # (cosine 0.01 decay)
        warmup_interval = min(1000, cfg.num_steps // 10)
        schedule = lambda x: min(
            1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
            0.01
            + 0.5
            * (1 - 0.01)
            * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )

        # (constant schedule)
        # warmup_interval = 1000
        # schedule = lambda x: (
        #     min(x, warmup_interval) / warmup_interval
        # )

        # (cosine 0.1 decay)
        # warmup_interval = min(2000, cfg.num_steps // 20)
        # schedule = lambda x: min(
        #     1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
        #     0.1
        #     + 0.5
        #     * (1 - 0.1)
        #     * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        # )

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

    checkpointer.save_single_file(cfg.num_steps, model)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
