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
    if not cfg.mup:
        raise ValueError("Must have mup=True")
    if cfg.tie_embeddings:
        raise ValueError("Tied Embedding-LMHead weights not supported.")


# Modified from mamba-ssm:
# https://github.com/state-spaces/mamba/blob/442fab4b1fd5226c1b5939b37d91ede430b5d1ae/mamba_ssm/models/mixer_seq_simple.py?plain=1#L91
# Basically, just need to insert a scaling factors in various places.


def _apply_mup_init(model: MambaLMHeadModel, cfg: mup_config) -> None:
    """
    Apply mup init.

    """
    if not isinstance(model, MambaLMHeadModel):
        warnings.warn(
            f"Found model of type {model.__class__.__name__}, not MambaLMHeadModel. No op."
        )
        return
    mup_cfg_check(cfg)
    # Rescale the lm head weights, preserving init when mup_ratio=1
    if cfg.mup_simple_scaling_impl:
        print("Using simple scaling mup init")
        _simple_mup_scaling_impl(model, cfg)
    else:
        print("Using custom mup init")
        _custom_mup_init(model, cfg)


def _simple_mup_scaling_impl(model: MambaLMHeadModel, cfg: mup_config) -> None:
    """
    This is a very minimal implementation where we just approximate the Table 3 impl 2203.03466
    through direct rescalings of the default mamba-ssm weights.

    The default mamba-ssm init is roughly:
    1) Normal-init the embedding layer with mean=0, std=.02
    2) Default-init the nn.Linear layers: uniform distribution, mean=0, std=1/sqrt(3*in_features)

    2) includes the LM head. The Table 3 impl instead requires that the LM head layer have a
    1/in_features std, and so we multiply its values by (mup_base_d_model / d_model)**0.5, which
    gives the right scaling while preserving the weights at `mup_base_d_model == d_model`

    The nn.Linear layers remain default initialized, whereas a stricter mup implementation would
    re-init from a normal distribution. This seems close enough, and by not performing any re-inits,
    we ensure perfect agreement between mup and no-mup in the d_model -> mup_base_d_model limit.
    """
    with torch.no_grad():
        model.lm_head.weight.mul_(cfg.mup_ratio**0.5)


def _custom_mup_init(model: MambaLMHeadModel, cfg: mup_config) -> None:
    """
    WIP!

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

    # MambaLMHeadModel organization:
    # Embedding: MambaLMHeadModel.backbone.embedding
    # Blocks: MambaLMHeadModel.backbone.layers
    # LM Head: MambaLMHeadModel.lm_head

    nn.init.normal_(model.backbone.embedding.weight, std=cfg.mup_initializer_range)

    blocks = model.backbone.layers
    for block in blocks:
        assert isinstance(block.mixer, MHA), "MHA only for now"
        assert isinstance(block.mlp, GatedMLP), "GatedMLP only for now"
        for layer in zip(block.mlp.modules(), block.mixer.modules()):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=1 / layer.in_feature**0.5)

    nn.init.normal_(model.lm_head.weight, mean=0.0, std=cfg.mup_ratio / (cfg.d_model))


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
    input_params_and_biases.append(model.backbone.embedding.weight)

    output_params.append(model.lm_head.weight)

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

    mup_param_groups = _get_mup_param_groups(model)

    # Create a list with a dict for each individual param. Annoying, but makes switching between
    # equivalent mup impls easier.
    if cfg.optim == "adamw":
        input_factor, hidden_factor, output_factor = 1, cfg.mup_ratio, cfg.mup_ratio
    elif cfg.optim == "sgd":
        input_factor, hidden_factor, output_factor = 1 / cfg.mup_ratio, 1, cfg.mup_ratio
    else:
        ValueError(f"Unexected {cfg.optim=}")

    optim_iter = [
        {"params": mup_param_groups.input, "lr": cfg.learning_rate * input_factor},
        {"params": mup_param_groups.hidden, "lr": cfg.learning_rate * hidden_factor},
        {"params": mup_param_groups.output, "lr": cfg.learning_rate * output_factor},
    ]
    return optim_iter
