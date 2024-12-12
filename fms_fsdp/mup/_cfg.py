from dataclasses import dataclass, field
from typing import Any, Optional, Union

from fms_fsdp.utils.config_utils import update_config

_OPTIMS = ("adamw", "sgd")


@dataclass
class mup_config:
    # model
    d_model: int = 512
    d_intermediate: Optional[int] = None  # Will default to 4*width if not provided
    n_layer: int = 10
    head_dim: int = 128
    attn_cfg: dict[str, Any] = field(
        default_factory=dict
    )  # Populated with defaults if not provided
    initializer_cfg: dict[str, Any] = field(default_factory=dict)
    tie_embeddings: bool = False

    # mup
    mup: bool = False
    mup_base_d_model: Optional[int] = None
    # Use very simple scaling strategy?
    mup_simple_scaling_impl: bool = False
    # mup_init_kwargs, chosen to approximate the mamba-ssm init scheme
    mup_initializer_range: float = (
        0.02  # Sets the std of the embedding layer. Name from mamba-ssm
    )
    mup_rescale_prenorm_residual: bool = True
    mup_use_d_model_attn_scaling: bool = False
    # Make weight.normal_(std=mup_lin_std_scale/in_features ** 0.5) have same std as the default
    # nn.Linear init.
    mup_lin_std_scale: float = 1 / (3**0.5)

    # From Davis; currently unused.
    mup_emb_scale: Optional[float] = (
        None  # Probably the same thing as mup_initializer_range?
    )
    mup_head_scale: Optional[float] = None
    mup_a_f_skew: Optional[float] = None
    mup_attn_temp: Optional[float] = None
    mup_lr_dscale: Optional[float] = None

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

    # training spec
    batch_size: int = 2
    acc_steps: int = 1
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0
    seed: int = 2023
    optim: str = "adamw"
    beta_0: float = 0.9
    beta_1: float = 0.95
    weight_decay: float = 0.1

    # FSDP
    world_size: int = 1
    sharding_strategy: str = "fsdp"
    fsdp_activation_checkpointing: bool = False
    selective_checkpointing: Union[float, str] = 1  # percentage of blocks to apply ac
    mixed_precision: bool = True
    low_cpu_fsdp: bool = False

    # logging
    report_interval: int = 100
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: Optional[str] = None
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = False

    def __post_init__(self) -> None:
        num_heads, remainder = divmod(self.d_model, self.head_dim)
        if remainder:
            raise ValueError(f"{self.head_dim=} must divide {self.d_model} evenly")
        if not self.attn_cfg:
            self.attn_cfg = {
                "causal": True,
                "head_dim": self.head_dim,
                "num_heads": num_heads,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                # Apparently rotary_emb_dim = head_dim // 2 for mamba-ssm:
                "rotary_emb_dim": self.head_dim // 2,
            }
            if self.mup and self.mup_use_d_model_attn_scaling:
                # If we are not scaling up head_dim, but rather just adding more heads, then it
                # doesn't make much sense to use the ~1/head_dim softmax normalization, but we
                # include it as a non-default option.
                self.attn_cfg["softmax_scale"] = 1 / self.head_dim
        if self.mup and not self.mup_base_d_model:
            raise ValueError("mup can only be specified along with a base_width")
        if self.tracker and self.tracker != "wandb":
            raise ValueError("Only tracker in {None, 'wandb'} supported")
        if self.d_intermediate is None:
            self.d_intermediate = 4 * self.d_model
        if self.optim not in _OPTIMS:
            raise ValueError(f"{self.optim=} must be one of {_OPTIMS}")

    @property
    def mup_ratio(self) -> float:
        if not self.mup:
            raise ValueError("mup_ratio only defined when mup=True")
        assert self.mup_base_d_model is not None  # mypy
        return self.mup_base_d_model / self.d_model

    @property
    def n_residuals_per_layer(self) -> int:
        # From mamba-ssm
        using_mlp = self.d_intermediate != 0
        return 2 if using_mlp else 1


def create_wandb_run_id(cfg: mup_config) -> str:
    user_run_id = cfg.tracker_run_id or ""
    run_id = (
        user_run_id
        + ("_" if user_run_id else "")
        + f"n_layer-{cfg.n_layer}_d_model-{cfg.d_model}_lr-{cfg.learning_rate}"
        + f"_bsz-{cfg.batch_size}_acc-{cfg.acc_steps}_seq_len-{cfg.seq_length}_steps"
        + f"-{cfg.num_steps}_world_size-{cfg.world_size}"
    )
    if cfg.mup:
        run_id += f"_mup[base-{cfg.mup_base_d_model}]"
    return run_id


def get_cfg_from_kwargs(**kwargs) -> mup_config:
    # get configs
    cfg = mup_config()
    update_config(cfg, **kwargs)
    print(f"--> running with these configs {cfg}")
    return cfg
