import math
import os
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from open_instruct.dataset_transformation import (
    get_cached_dataset_tulu,
)
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoConfig, AutoTokenizer

from fms_fsdp import config
from fms_fsdp.utils.checkpointing_utils_sft import Checkpointer
from fms_fsdp.utils.config_utils_sft import get_model_config, update_config
from fms_fsdp.utils.dataloader_utils_sft import (
    InfiniteCPBatchingIter,
    PretokenizedCollator,
    get_chat_template,
)
from fms_fsdp.utils.train_utils_sft import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
)


def parse_args(x):
    if isinstance(x, str):
        return [item.strip() for item in x.split(",")]
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (int, float, complex)):
        return [x]
    raise ValueError(f"arg input {x} cannot be parsed.")


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
    setup(cfg)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()
    if triton_cache_dir := os.getenv("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = os.path.join(
            triton_cache_dir, "fms_fsdp", str(local_rank)
        )
    else:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(
            Path.home(), ".cache", "triton", "fms_fsdp", str(local_rank)
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
    def get_1D_world_mesh(world_size: int, prefix: str) -> DeviceMesh:
        mesh = dist.device_mesh.init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("prefix",)
        )
        return mesh

    def get_2D_world_mesh(world_size: int, inner_size: int, prefix: str) -> DeviceMesh:
        assert world_size % inner_size == 0
        mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (world_size // inner_size, inner_size),
            mesh_dim_names=(f"{prefix}_outer", f"{prefix}_inner"),
        )
        return mesh

    # NOTE: @goon - for some reason, just creating a single 1D or 2D mesh and using slices of that
    # as appropriate seems to give much less stable behavior than making separate CP and FSDP
    # meshes.
    if cfg.cp:
        cp_degree = cfg.cp_degree or torch.cuda.device_count()
        if cp_degree == world_size:
            cp_mesh = get_1D_world_mesh(world_size, prefix="cp")
            dp_mesh = None
            dp_rank = 0
            cp_rank = cp_mesh.get_local_rank()
        else:
            two_d_mesh = get_2D_world_mesh(world_size, cp_degree, prefix="cp")
            cp_mesh = two_d_mesh["cp_inner"]
            dp_mesh = two_d_mesh["cp_outer"]
            cp_rank = two_d_mesh["cp_inner"].get_local_rank()
            dp_rank = two_d_mesh["cp_outer"].get_local_rank()
    else:
        cp_mesh = None
        cp_degree = 1
        cp_rank = 0
        dp_rank = rank
        dp_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp",),
        )
    dp_degree = world_size // cp_degree
    print(f"Rank assignments: {rank=}, {dp_rank=}, {cp_rank=}")

    if cfg.sharding_strategy == "fsdp":
        fsdp_mesh = get_1D_world_mesh(world_size, prefix="fsdp")
    elif cfg.sharding_strategy == "hsdp":
        fsdp_mesh = get_2D_world_mesh(
            world_size, torch.cuda.device_count(), prefix="fsdp"
        )
    else:
        fsdp_mesh = None

    # Init meshes
    if not rank:
        print("Initializing meshes with barriers:")
    for mesh in (cp_mesh, dp_mesh):
        if mesh is not None:
            dist.barrier(mesh.get_group())

    if fsdp_mesh.ndim == 1:
        dist.barrier(fsdp_mesh.get_group())
    else:
        dist.barrier(fsdp_mesh["fsdp_inner"].get_group())
        dist.barrier(fsdp_mesh["fsdp_outer"].get_group())

    torch.cuda.synchronize()
    if not rank:
        print("Done mesh init.")

    # get model
    config_data = get_model_config(cfg.model_variant)
    mamba_config = MambaConfig(**config_data)
    if not rank:
        print(f"{mamba_config=}")

    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = MambaLMHeadModel(
                mamba_config,
                cp_mesh=cp_mesh if cfg.cp else None,
                cp_mamba_impl=cfg.cp_mamba_impl if cfg.cp else None,
                cp_attn_impl=cfg.cp_attn_impl if cfg.cp else None,
                cp_mamba_recompute=cfg.cp_mamba_recompute if cfg.cp else None,
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
    tokenizer.chat_template = get_chat_template(cfg.chat_template_name)
    if rank == 0:
        print(f"{tokenizer.chat_template=}")

    # Assumption: all datasets are pretokenized already and saved with HF's Dataset.save_to_disk
    # See tests/sft/tokenize_dataset.py
    dataset_config_hashes = [h.strip() for h in parse_args(cfg.dataset_config_hashes)]
    dataset_local_cache_dir = [
        str(d).strip() for d in parse_args(cfg.dataset_local_cache_dir)
    ]
    if len(dataset_config_hashes) != len(dataset_local_cache_dir):
        if len(dataset_local_cache_dir) != 1:
            raise ValueError(
                f"Either {len(dataset_local_cache_dir)=} must equal {len(dataset_config_hashes)=} or 1"
            )
        dataset_local_cache_dir = len(dataset_config_hashes) * dataset_local_cache_dir

    train_dataset_list = []
    for dset_hash, dset_cache_dir in zip(
        dataset_config_hashes, dataset_local_cache_dir
    ):
        # NOTE: @goon - None fields aren't using the proper types, but aren't used when loading
        # from the cache.
        dataset = get_cached_dataset_tulu(
            dataset_mixer_list=None,
            dataset_mixer_list_splits=None,
            tc=None,
            dataset_transform_fn=None,
            transform_fn_args=None,
            target_columns=None,
            dataset_cache_mode="local",
            dataset_config_hash=dset_hash,
            hf_entity=None,
            dataset_local_cache_dir=dset_cache_dir,
            dataset_skip_cache=False,
            keep_in_memory=False,
        )
        dataset = dataset.shuffle(seed=cfg.seed)
        dataset.set_format(type="pt")
        train_dataset_list.append(dataset)

    dataset_lens = [len(d) for d in train_dataset_list]
    if not rank:
        print(
            f"Train datasets loaded with {sum(len(td) for td in train_dataset_list)} total examples"
        )

    samplers = [
        DistributedSampler(
            td,
            num_replicas=dp_degree,
            rank=dp_rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=False,
        )
        for td in train_dataset_list
    ]
    train_loader_list = [
        DataLoader(
            td,
            sampler=sampler,
            collate_fn=PretokenizedCollator(),
            # NOTE: @goon -  batch size is intentionally one. InfiniteCPBatchingIter handles forming
            # batches of the appropriate size.
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        for td, sampler in zip(train_dataset_list, samplers)
    ]

    train_loader = InfiniteCPBatchingIter(
        train_loader_list,
        weights=parse_args(cfg.weights),
        batch_size=cfg.batch_size,
        seq_length=cfg.seq_length,
        max_out_tokens=cfg.max_out_tokens,
        cp_degree=cp_degree,
        cp_rank=cp_rank,
        dataset_lens=dataset_lens,
        pad_id=cfg.pad_id,
        separator_id=cfg.separator_id,
        seed=cfg.seed,
        naive_padding_free=cfg.naive_padding_free,
        weight_by=cfg.weight_by,
    )
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
        betas=(cfg.adam_beta_0, cfg.adam_beta_1),
        weight_decay=cfg.weight_decay,
    )

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank
    )

    if cfg.hf_cfg_path is not None:
        hf_cfg_path = cfg.hf_cfg_path
    elif (Path(cfg.ckpt_load_path) / "config.json").exists():
        hf_cfg_path = cfg.ckpt_load_path
    else:
        raise ValueError(
            "Please either provide a hf_cfg_path or point the ckpt_load_path to a HF ckpt dir"
        )
    hf_config = AutoConfig.from_pretrained(hf_cfg_path)
    if getattr(hf_config, "embedding_multiplier", 1.0) != 1.0:
        raise NotImplementedError
    if getattr(hf_config, "residual_multiplier", 1.0) != 1.0:
        raise NotImplementedError

    ckpt_load_path = Path(cfg.ckpt_load_path)
    is_hf_ckpt_path = (ckpt_load_path / "config.json").exists()
    if ckpt_load_path.is_file() or is_hf_ckpt_path:
        ckpt_load_path_str = cfg.ckpt_load_path
    else:
        ckpt_load_path_str = os.path.join(cfg.ckpt_load_path, "checkpoints/")

    model, optimizer, _, start_step, tokens_seen, pred_tokens_seen, is_resuming = (
        checkpointer.load(
            model,
            optimizer,
            None,
            hf_config,
            path=cfg.ckpt_load_path,
            strict=True,
        )
    )
    if not is_resuming:
        start_step = 0
        # Override loaded optim hyperparams with the current values
        for g in optimizer.param_groups:
            g["initial_lr"] = cfg.learning_rate

    # LR schedule
    # linear decay for annealing
    if cfg.training_stage == "annealing":
        schedule = lambda x: min(
            1 - (1 - min(x, cfg.warmup_interval) / cfg.warmup_interval) ** 2,
            1
            - (1 - cfg.final_lr_ratio)
            * (x - cfg.warmup_interval)
            / (cfg.num_steps - cfg.warmup_interval),
        )
    elif cfg.training_stage == "constant":
        schedule = lambda x: (min(x, cfg.warmup_interval) / cfg.warmup_interval)
    elif cfg.training_stage == "cosine":
        # cosine decay
        schedule = lambda x: min(
            1 - (1 - min(x, cfg.warmup_interval) / cfg.warmup_interval) ** 2,
            0.1
            + 0.5
            * (1 - 0.1)
            * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
        )
    else:
        raise ValueError(f"{cfg.training_stage=} not in (annealing, constant, cosine)")

    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # Data scheduler. Returns the current max seq_len (not like the LR scheduler, which returns
    # a number between zero and 1)
    if cfg.data_schedule == "constant":
        data_schedule = lambda x: cfg.seq_length
    elif cfg.data_schedule == "linear":
        data_schedule = (
            lambda x: (x - 1) * cfg.seq_length / (cfg.num_steps - 1)
            + cfg.min_seq_length
        )
    elif cfg.data_schedule == "quadratic":
        data_schedule = (
            lambda x: (x - 1) ** 2 * cfg.seq_length / (cfg.num_steps - 1) ** 2
            + cfg.min_seq_length
        )
    else:
        raise ValueError(f"{cfg.data_schedule=} not in (constant, linear, quadratic)")

    # Skip previous batches
    if start_step > 0:
        if not rank:
            print("Skipping previous data after restart!")
        n_batches_to_skip = cfg.grad_accum_steps * start_step
        for batch_idx in range(1, n_batches_to_skip + 1):
            step_idx = (batch_idx + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
            train_loader.seq_length = data_schedule(step_idx)
            next(train_loader)
        if not rank:
            print("Skipping done!")
    # profiler
    profiler = get_profiler(cfg, rank)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    train(
        cfg,
        mamba_config,
        model,
        tokenizer,
        local_rank,
        rank,
        cp_degree,
        dp_mesh,
        train_loader,
        optimizer,
        scheduler,
        profiler,
        checkpointer,
        start_step,
        tokens_seen,
        pred_tokens_seen,
        hf_config,
        dataset_lens,
        data_schedule,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
