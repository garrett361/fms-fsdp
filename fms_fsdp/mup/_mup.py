import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP

from fms_fsdp.mup._cfg import mup_config

"""
Basic mup implementation following Table 3 of 2203.03466. Specific to MambaLMHeadModel.
"""


def mup_cfg_check(cfg: MambaConfig) -> None:
    if cfg.tie_embeddings:
        raise ValueError("Tied Embedding-LMHead weights not supported.")


# Modified from mamba-ssm:
# https://github.com/state-spaces/mamba/blob/442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/models/mixer_seq_simple.py?plain=1#L91
def _init_weights(
    module: nn.Module,
    n_layer: int,
    cfg: mup_config,
) -> None:
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=cfg.mup_initializer_range)

    if cfg.mup_rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(cfg.mup_n_residuals_per_layer * n_layer)


def apply_mup_init(model: MambaLMHeadModel, cfg: mup_config) -> None:
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
        warnings.warn(
            f"Found model of type {model.__class__.__name__}, not MambaLMHeadModel. No op."
        )
        return
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


@dataclass
class MupParam:
    param: nn.Parameter
    fan_in: int
    fan_out: int


@dataclass
class MupParamGroups:
    input: list[MupParam]
    hidden: list[MupParam]
    output: list[MupParam]


def _get_mup_param_groups(model: MambaLMHeadModel) -> MupParamGroups:
    # Nomenclature of 2203.03466
    input_params_and_biases: list[MupParam] = []
    hidden_params: list[MupParam] = []
    output_params: list[MupParam] = []

    # Embedding and lm head are in- and output-params respectively
    emb = model.backbone.embedding
    # I think the fan_in for the embedding weight is 1?
    input_params_and_biases.append(
        MupParam(emb.weight, fan_in=1, fan_out=emb.embedding_dim)
    )

    lm_head = model.lm_head
    output_params.append(
        MupParam(
            lm_head.weight, fan_in=lm_head.in_features, fan_out=lm_head.out_features
        )
    )

    # There is also a final layer norm in the backbone
    for p_name, p in model.backbone.norm_f.named_parameters():
        assert len(p.shape) == 1, f"{p_name=}, {len(p.shape)=}"
        input_params_and_biases.append(MupParam(p, fan_in=1, fan_out=p.shape[0]))

    # Everything else is blocks
    blocks = model.backbone.layers
    for block in blocks:
        for module in block.modules():
            if isinstance(module, nn.Linear):
                hidden_params.append(
                    MupParam(
                        module.weight,
                        fan_in=module.in_features,
                        fan_out=module.out_features,
                    )
                )
                if module.bias is not None:
                    input_params_and_biases.append(
                        MupParam(
                            module.bias,
                            fan_in=1,
                            fan_out=module.out_features,
                        )
                    )
            else:
                # Don't recurse, otherwise we will double-count the Linear layers above
                # Assumption: everything else is a layer-norm type layer
                for p_name, p in module.named_parameters(recurse=False):
                    assert len(p.shape) == 1, f"{p_name=}, {len(p.shape)=}"
                    input_params_and_biases.append(
                        MupParam(p, fan_in=1, fan_out=p.shape[0])
                    )

    total_params = len(list(model.parameters()))
    params_accounted_for = (
        len(input_params_and_biases) + len(hidden_params) + len(output_params)
    )
    assert (
        total_params == params_accounted_for
    ), f"{total_params=}, {params_accounted_for}"

    return MupParamGroups(
        input=input_params_and_biases, hidden=hidden_params, output=output_params
    )


def get_mup_optim_iter(
    model: MambaLMHeadModel,
    cfg: mup_config,
) -> list[dict]:
    """
    Get the per-weight learning rates for mup.
    """
    expected_optims = ("adamw", "sgd")
    if cfg.optim not in expected_optims:
        raise ValueError(f"Expected {cfg.optim=} to be in {expected_optims}")
    if cfg.optim == "sgd":
        raise NotImplementedError("Just adamw for now.")

    mup_param_groups = _get_mup_param_groups(model)

    # Create a list with a dict for each individual param. Annoying, but makes switching between
    # equivalent mup impls easier.

    optim_iter = [
        {"params": [mp.param], "lr": cfg.learning_rate} for mp in mup_param_groups.input
    ]
    optim_iter.extend(
        [
            {
                "params": [mp.param],
                "lr": cfg.learning_rate * cfg.mup_base_d_model / (cfg.d_model or mp.fan_in),
            }
            for mp in mup_param_groups.hidden
        ]
    )
    optim_iter.extend(
        [
            {
                "params": [mp.param],
                "lr": cfg.learning_rate * cfg.mup_base_d_model / (cfg.d_model or mp.fan_in),
            }
            for mp in mup_param_groups.output
        ]
    )
    return optim_iter
