from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def apply_mup_init(
    model: MambaLMHeadModel,
    mup_emb_scale: float = 0,
    mup_head_scale: float = 0,
    mup_a_f_skew: float = 0,
    mup_attn_temp: float = 0,
    mup_lr_dscale: float = 0,
) -> None:
    if not isinstance(model, MambaLMHeadModel):
        raise ValueError(
            f"mup only implemented for MambaLMHeadModel classes, not {model.__class__.__name__}"
        )
    if model.config.tie_embeddings:
        raise ValueError("Tied Embedding-LMHead weights not supported.")
    # MambaLMHeadModel organization:
    # Embedding: MambaLMHeadModel.backbone.embedding
    # Blocks: MambaLMHeadModel.backbone.layers
    # LM Head: MambaLMHeadModel.lm_head

    assert model.backbone.embedding is not None
    for layer_idx, block in enumerate(model.backbone.layers):
        print(f"Found {layer_idx=}: {block=}")
    assert model.lm_head is not None
