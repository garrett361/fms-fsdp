from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

"""
Basic mup implementation following Table 3 of 2203.03466. Specific to MambaLMHeadModel.
"""


def mup_cfg_check(cfg: MambaConfig) -> None:
    if cfg.tie_embeddings:
        raise ValueError("Tied Embedding-LMHead weights not supported.")


def apply_mup_init(
    model: MambaLMHeadModel,
    mup_emb_scale: float = 0,
    mup_head_scale: float = 0,
    mup_a_f_skew: float = 0,
    mup_attn_temp: float = 0,
    mup_lr_dscale: float = 0,
) -> None:
    """
    Apply mup init.

    Notes:

    mamba-ssm already does some init:

    - https://github.com/state-spaces/mamba/blob/442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/models/mixer_seq_simple.py#L175-L182
    - https://github.com/state-spaces/mamba/blob/442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/models/mixer_seq_simple.py#L86-L86

    This does the following by default:
    - Zeros out all Linear.bias terms except some which are important for mamba
    - Forces the nn.Embedding layer to have 0.02 init
    - Rescales some Linear weights following some prescription GPT-2 for residuals
    """
    if not isinstance(model, MambaLMHeadModel):
        raise ValueError(
            f"mup only implemented for MambaLMHeadModel classes, not {model.__class__.__name__}"
        )
    cfg = model.config
    mup_cfg_check(cfg)

    # MambaLMHeadModel organization:
    # Embedding: MambaLMHeadModel.backbone.embedding
    # Blocks: MambaLMHeadModel.backbone.layers
    # LM Head: MambaLMHeadModel.lm_head

    # The embedding layer is an nn.Embedding which already performs unit-normal init and zeros out
    # the padding entry, if needed.
    # reset_parameters() perform unit-normal init and zeros out the padding entry, if applicable.
    model.backbone.embedding.reset_parameters()

    blocks = model.backbone.layers
    for block in blocks:
        assert isinstance(block.mixer, MHA), "MHA only for now"
        assert isinstance(block.mlp, GatedMLP), "GatedMLP only for now"
        for layer in zip(block.mlp.modules(), block.mixer.modules()):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1 / layer.in_feature**0.5)

    nn.init.normal_(model.lm_head.weight, mean=0.0, std=1 / cfg.d_model)
