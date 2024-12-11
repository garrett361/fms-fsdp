from dataclasses import asdict
from inspect import signature

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.mup._cfg import mup_config
from fms_fsdp.mup._mup import _apply_mup_init


def get_mamba_cfg_from_mup_cfg(cfg: mup_config) -> MambaConfig:
    filtered_kwargs = {
        k: v for k, v in asdict(cfg).items() if k in signature(MambaConfig).parameters
    }
    # Force the model to be transformer-only:
    filtered_kwargs["attn_layer_idx"] = list(range(cfg.n_layer))
    return MambaConfig(**filtered_kwargs)


def get_transformer(cfg: mup_config, device: str = "cuda") -> MambaLMHeadModel:
    """
    Creates a transformer-only model. Applies mup-init, if applicable.
    """
    # The mup paper discusses scaling up the attn_cfg.softmax_scale property, but that only seems
    # relevant if we are scaling up head_dim, which we are not currently doing.

    mamba_cfg = get_mamba_cfg_from_mup_cfg(cfg)
    print(f"Building model with {mamba_cfg=}")
    model = MambaLMHeadModel(mamba_cfg, device=device)
    if cfg.mup:
        _apply_mup_init(model, cfg)
    return model
