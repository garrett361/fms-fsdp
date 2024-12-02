from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from dataclasses import dataclass

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


def get_mup_param_groups(model: MambaLMHeadModel) -> MupParamGroups:
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
    lr: float,
    optim_type: str = "adam",
) -> list[dict]:
    expected_optim_types = ("adam", "sgd")
    if optim_type not in expected_optim_types:
        raise ValueError(f"Expected {optim_type=} to be in {expected_optim_types}")
    if optim_type == "sgd":
        raise NotImplementedError("Just adam for now.")

    mup_param_groups = get_mup_param_groups(model)

    # Create a list with a dict for each individual param. Annoying, but makes switching between
    # equivalent mup impls easier.
    optim_iter = [{"params": [mp.param], "lr": lr} for mp in mup_param_groups.input]
    optim_iter.extend(
        [{"params": [mp.param], "lr": lr / mp.fan_in} for mp in mup_param_groups.hidden]
    )
    optim_iter.extend(
        [{"params": [mp.param], "lr": lr / mp.fan_in} for mp in mup_param_groups.output]
    )
    return optim_iter
