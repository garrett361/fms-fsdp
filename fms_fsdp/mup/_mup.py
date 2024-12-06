import math
import warnings
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.mup._cfg import mup_config

"""
Basic mup implementation following Table 3 of 2203.03466. Specific to MambaLMHeadModel.
"""


def mup_cfg_check(cfg: MambaConfig) -> None:
    if not cfg.mup:
        raise ValueError("Must have mup=True")
    if cfg.tie_embeddings:
        raise ValueError("Tied Embedding-LMHead weights not supported.")


# Modified from mamba-ssm:
# https://github.com/state-spaces/mamba/blob/442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/models/mixer_seq_simple.py?plain=1#L91
# Basically, just need to insert a scaling factors in various places.


# NOTE: @goon - whether cfg.mup_ratio factors are inserted depends on which mup impl from 2203.03466
# is used. Currently following Table 3, which requires no factors in _init_weights (but see the
# cfg.mup_ratio in _apply_mup_init applied to the LM head.
def _init_weights(
    module: nn.Module,
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
                    p.div_(math.sqrt(cfg.n_residuals_per_layer * cfg.n_layer))


def _apply_mup_init(model: MambaLMHeadModel, cfg: mup_config) -> None:
    """
    Apply mup init.

    Notes:

    mamba-ssm already does some init:

    - https://github.com/state-spaces/mamba/blob/442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/models/mixer_seq_simple.py#L175-L182
    - https://github.com/state-spaces/mamba/blob/442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/models/mixer_seq_simple.py#L86-L86

    This reapplies the same code, but with mup scaling factors inserted.
    """
    if not isinstance(model, MambaLMHeadModel):
        warnings.warn(
            f"Found model of type {model.__class__.__name__}, not MambaLMHeadModel. No op."
        )
        return
    mup_cfg_check(cfg)
    model.apply(partial(_init_weights, cfg=cfg))
    # Rescale the lm head weights, preserving init when mup_ratio=1
    with torch.no_grad():
        model.lm_head.weight.mul_(cfg.mup_ratio)


@dataclass
class MupParamGroups:
    input: list[nn.Parameter]
    hidden: list[nn.Parameter]
    output: list[nn.Parameter]


def _get_mup_param_groups(model: MambaLMHeadModel) -> MupParamGroups:
    # Nomenclature of 2203.03466
    input_params_and_biases: list[nn.Parameter] = []
    hidden_params: list[nn.Parameter] = []
    output_params: list[nn.Parameter] = []

    # Embedding and lm head are in- and output-params respectively
    emb = model.backbone.embedding
    # I think the fan_in for the embedding weight is 1?
    input_params_and_biases.append(emb.weight)

    lm_head = model.lm_head
    output_params.append(lm_head.weight)

    # There is also a final layer norm in the backbone
    for p_name, p in model.backbone.norm_f.named_parameters():
        assert len(p.shape) == 1, f"{p_name=}, {len(p.shape)=}"
        input_params_and_biases.append(p)

    # Everything else is blocks
    blocks = model.backbone.layers
    for block in blocks:
        for module in block.modules():
            if isinstance(module, nn.Linear):
                hidden_params.append(module.weight)
                if module.bias is not None:
                    input_params_and_biases.append(module.bias)
            else:
                # Don't recurse, otherwise we will double-count the Linear layers above
                # Assumption: everything else is a layer-norm type layer
                for p_name, p in module.named_parameters(recurse=False):
                    assert len(p.shape) == 1, f"{p_name=}, {len(p.shape)=}"
                    input_params_and_biases.append(p)

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
        {"params": [p], "lr": cfg.learning_rate} for p in mup_param_groups.input
    ]
    optim_iter.extend(
        [
            {"params": [p], "lr": cfg.learning_rate * cfg.mup_ratio}
            for p in mup_param_groups.hidden
        ]
    )
    optim_iter.extend(
        [
            {"params": [p], "lr": cfg.learning_rate * cfg.mup_ratio}
            for p in mup_param_groups.output
        ]
    )
    return optim_iter
