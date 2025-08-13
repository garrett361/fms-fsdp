from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class train_config:
    # model
    model_variant: str = "7b"
    ckpt_load_path: str = "/fsx/output/ckpt"
    ckpt_save_path: str = "/fsx/output/ckpt"

    # dataset and dataloader
    sanity_print_toks: bool = False
    tokenizer_path: str = "/fsx/tokenizer"
    dataset_config_hashes: str = "" # Comma separated list of hashes
    weights: str = "" # Comma separated weights for the datasets. Must match the number of hashes.
    dataset_local_cache_dir: str = ""
    seq_length: int = 4096
    vocab_size: int = 32000
    num_workers: int = 1
    weight_by: Literal["example", "token", "pred_token"] = "example"

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
    z_loss: Optional[float] = None
    grad_acc_steps: int = 1
    weight_decay: float = 0.1
    adam_beta_0: float = 0.9
    adam_beta_1: float = 0.95

    # profiling
    use_profiler: bool = False
    profiler_rank0_only: bool = True

    # logging
    report_interval: int = 100
    checkpoint_interval: int = 10000
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = True

    # context parallel
    cp: bool = False
    cp_mamba_impl: str = "allgather"  # "allgather" or "serial"
    cp_mamba_recompute: bool = False
    cp_attn_impl: str = "zigzag"  # "zigzag" or "ring"
    cp_degree: Optional[int] = None

    # SFT
    chat_template_name: str = "tulu"
    sft_loss_type: str = "sum"  # sum or mean. The `reduction` arg for F.cross_entropy
    final_lr_ratio: float = 0.1  # ratio of initial to final lr during sft
    sft_warmup_fraction: float = 0.05  # Fraction of SFT steps to perform LR warmup on
    separator_id: int = -100
    pad_id: int = 0
    naive_padding_free: bool = False
    pg_timeout_s: int = (
        10 * 60
    )  # 10 min; checkpointing is taking a long time for some reason. TODO: @goon - investigate
    pin_memory: bool = True

    # HF ckpt
    hf_cfg_path: Optional[str] = None

    # debug
    skip_ckpt: bool = False
    skip_optim_step: bool = False
