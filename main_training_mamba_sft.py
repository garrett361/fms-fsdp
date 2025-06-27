import math
import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from warnings import warn

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils_sft import Checkpointer
from fms_fsdp.utils.config_utils import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils import (
    ChatTokenizerCollatorCPCollator,
    CPDataCollator,
    get_infinite_iter,
)
from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES, encode_sft_example
from fms_fsdp.utils.train_utils_sft import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)


@contextmanager
def local_rank_zero_first(rank):
    is_local_rank_zero = rank % torch.cuda.device_count() == 0
    if not is_local_rank_zero:
        dist.barrier()

    yield
    if is_local_rank_zero:
        dist.barrier()


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
    dist.barrier()

    # get policy. NOTE: @goon - overriding {wrapping_policy, param_init_fn} below
    block = Block
    (
        mixed_precision_policy,
        _,
        sharding_strategy_policy,
        apply_selective_ac,
        _,  # NOTE: @goon - We'll override param_init_fn for mamba below
    ) = get_policies(cfg, rank, block)
    if cfg.low_cpu_fsdp:

        def param_init_fn(module):
            module.to_empty(device=torch.cuda.current_device())
            if hasattr(module, "reset_parameters"):
                with torch.no_grad():
                    module.reset_parameters()
    else:
        param_init_fn = None

    # Meshes for FSDP and CP. NOTE: @goon - Getting hangs and/or OOMs if I don't explicitly specify
    # the FSDP mesh when using 4+ nodes with HSDP + in-node-CP.
    def get_1D_world_mesh(world_size: int) -> DeviceMesh:
        mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))
        return mesh

    def get_2D_world_mesh(world_size: int, inner_size: int) -> DeviceMesh:
        assert world_size % inner_size == 0
        mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (world_size // inner_size, inner_size),
            mesh_dim_names=("outer", "inner"),
        )
        return mesh

    # NOTE: @goon - for some reason, just creating a single 1D or 2D mesh and using slices of that
    # as appropriate seems to give much less stable behavior than making separate CP and FSDP
    # meshes.
    if cfg.cp:
        cp_degree = cfg.cp_degree or torch.cuda.device_count()
        if cp_degree == world_size:
            cp_mesh = get_1D_world_mesh(world_size)
            dp_rank = 0
            cp_rank = cp_mesh.get_local_rank()
        else:
            two_d_mesh = get_2D_world_mesh(world_size, cp_degree)
            cp_mesh = two_d_mesh["inner"]
            dp_rank = two_d_mesh["outer"].get_local_rank()
            cp_rank = two_d_mesh["inner"].get_local_rank()
    else:
        cp_mesh = None
        cp_degree = 1
        cp_rank = 0
        dp_rank = rank
    dp_degree = world_size // cp_degree

    if cfg.sharding_strategy == "fsdp":
        fsdp_mesh = get_1D_world_mesh(world_size)
    elif cfg.sharding_strategy == "hsdp":
        fsdp_mesh = get_2D_world_mesh(world_size, torch.cuda.device_count())
    else:
        fsdp_mesh = None

    # get model
    config_data = get_model_config(cfg.model_variant)
    mamba_config = MambaConfig(**config_data)

    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = MambaLMHeadModel(
                mamba_config,
                cp_mesh=cp_mesh if cfg.cp else None,
                cp_mamba_impl=cfg.cp_mamba_impl if cfg.cp else None,
                cp_attn_impl=cfg.cp_attn_impl if cfg.cp else None,
            )
    else:
        model = MambaLMHeadModel(
            mamba_config,
            cp_mesh=cp_mesh if cfg.cp else None,
            cp_mamba_impl=cfg.cp_mamba_impl if cfg.cp else None,
            cp_attn_impl=cfg.cp_attn_impl if cfg.cp else None,
        )

    def lambda_fn(module: nn.Module):
        return isinstance(module, (Block, nn.Embedding)) or module is model.lm_head

    wrapping_policy = CustomPolicy(lambda_fn)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenizer.chat_template = CHAT_TEMPLATES[cfg.chat_template_name]

    with local_rank_zero_first(rank):
        train_dataset = load_dataset(
            "parquet", data_dir=cfg.data_path, num_proc=cfg.num_workers
        )["train"]
        # For sanity checking:
        if cfg._n_examples:
            warn(f"only using {cfg._n_examples=} for sanity checking!", stacklevel=1)
            train_dataset = train_dataset.select(range(cfg._n_examples))
        if not rank:
            print("Dataset loaded")

        print(f"Rank assignments: {rank=}, {dp_rank=}, {cp_rank=}")
        if not cfg.use_dummy_dataset:
            # NOTE: @goon - open-instruct pre-maps the training example around this point, but this can
            # takes a long time, so we also give the option to tokenize on the fly
            if cfg.tokenize_on_fly:
                sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=dp_degree,
                    rank=dp_rank,
                    shuffle=True,
                    seed=cfg.seed,
                    drop_last=False,
                )
                collate_fn = ChatTokenizerCollatorCPCollator(
                    tokenizer=tokenizer,
                    max_seq_length=cfg.seq_length,
                    cp_degree=cp_degree,
                    cp_rank=cp_rank,
                    pad_id=0,
                    separator_id=-100,
                )
            else:
                assert tokenizer.is_fast
                train_dataset = train_dataset.map(
                    partial(
                        encode_sft_example,
                        tokenizer=tokenizer,
                        max_seq_length=cfg.seq_length,
                    ),
                    batched=False,
                    num_proc=torch.cuda.device_count() * cfg.num_workers,
                    remove_columns=[
                        name
                        for name in train_dataset.column_names
                        if name not in ["input_ids", "labels"]
                    ],
                    desc="Tokenizing and reformatting instruction data",
                )
                train_dataset.set_format(type="pt")
                train_dataset = train_dataset.filter(
                    lambda example: (example["labels"] != -100).any()
                )
                collate_fn = CPDataCollator(
                    cp_degree=cp_degree,
                    cp_rank=cp_rank,
                    pad_id=0,
                    separator_id=-100,
                )

            train_loader = DataLoader(
                train_dataset,
                sampler=sampler,
                collate_fn=collate_fn,
                batch_size=cfg.batch_size,
            )
            train_loader = get_infinite_iter(train_loader)
        else:
            raise ValueError("This script assumes no dummy loader is used")
    if rank == 0:
        print("Datasets constructed!")

    # FSDP
    model = FSDP(
        model,
        device_mesh=fsdp_mesh,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=param_init_fn,
    )
    if rank == 0:
        print(model)

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

    # Optimizer
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
    model, optimizer, _, start_step, tokens_seen, pred_tokens_seen, is_resuming = (
        checkpointer.load(
            model,
            optimizer,
            None,
            path=os.path.join(cfg.ckpt_load_path, "checkpoints/")
            if not os.path.isfile(cfg.ckpt_load_path)
            else cfg.ckpt_load_path,
            strict=False,
        )
    )
    if not is_resuming:
        start_step = 0
        # Override loaded optim hyperparams with the current values
        for g in optimizer.param_groups:
            g["initial_lr"] = cfg.learning_rate

    # Skip previous batches
    if start_step > 0:
        # TODO: @goon - check off by one
        for _ in range(cfg.grad_acc_steps * start_step):
            next(train_loader)

    # LR schedule
    # linear decay for annealing
    assert cfg.training_stage == "annealing", "SFT expects an annealing scheduler"
    if cfg.training_stage == "annealing":
        warmup_interval = cfg.num_steps * cfg.sft_warmup_fraction
        schedule = lambda x: min(
            1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
            1
            - (1 - cfg.final_lr_ratio)
            * (x - warmup_interval)
            / (cfg.num_steps - warmup_interval),
        )
    else:
        # cosine decay
        warmup_interval = min(2000, cfg.num_steps // 20)
        schedule = lambda x: min(
            1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
            0.1
            + 0.5
            * (1 - 0.1)
            * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )

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
        pred_tokens_seen,
    )

    checkpointer.save_single_file(cfg.num_steps, model)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
