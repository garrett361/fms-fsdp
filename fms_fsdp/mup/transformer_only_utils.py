from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from fms_fsdp.mup.mup_mamba import apply_mup_init


def get_transformer_and_config(
    width: int,
    n_layer: int = 10,
    vocab_size: int = 128256,
    head_dim: int = 128,
    device: str = "cuda",
    mup: bool = False,
) -> tuple[MambaLMHeadModel, MambaConfig]:
    """
    Get small transformer models. Config based on Cerebras-GPT 111M, roughly.
    """
    num_heads, remainder = divmod(width, head_dim)
    if remainder:
        raise ValueError(f"Choose {width=} divisible by {head_dim=}.")
    attn_cfg = {
        "causal": True,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": head_dim // 2,  # Apparently correct for mamba-ssm
    }
    if mup:
        attn_cfg["softmax_scale"] = head_dim

    config = MambaConfig(
        d_model=width,
        d_intermediate=4 * width,
        n_layer=n_layer,
        attn_layer_idx=list(range(n_layer)),  # Transformer-only blocks
        vocab_size=vocab_size,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )
    model = MambaLMHeadModel(config, device=device)
    if mup:
        print("Applying mup param init")
        apply_mup_init(model)
    return model, config
