from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from fms_fsdp.mup._cfg import mup_config
from fms_fsdp.mup._mup import apply_mup_init


def get_transformer(cfg: mup_config, device: str = "cuda") -> MambaLMHeadModel:
    # The mup paper discusses scaling up the attn_cfg.softmax_scale property, but that is only
    # relevant if we are scaling up head_dim, which we are not currently doing.
    # if cfg.mup:
    #     cfg.attn_cfg["softmax_scale"] = cfg.head_dim

    config = MambaConfig(
        d_model=cfg.d_model,
        d_intermediate=cfg.d_intermediate,
        n_layer=cfg.n_layer,
        attn_layer_idx=list(range(cfg.n_layer)),  # Transformer-only blocks
        vocab_size=cfg.vocab_size,
        attn_cfg=cfg.attn_cfg,
        tie_embeddings=cfg.tie_embeddings,
    )
    model = MambaLMHeadModel(config, device=device)
    if cfg.mup:
        print("Applying mup param init")
        apply_mup_init(model, cfg)
    return model
