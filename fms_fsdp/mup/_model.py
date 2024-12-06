from dataclasses import asdict
from inspect import signature
import torch

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.mup._cfg import mup_config
from fms_fsdp.mup._mup import apply_mup_init


def get_transformer(cfg: mup_config, device: str = "cuda") -> MambaLMHeadModel:
    """
    Creates a transformer-only model. Applies mup-init, if applicable.
    """
    # The mup paper discusses scaling up the attn_cfg.softmax_scale property, but that is only
    # relevant if we are scaling up head_dim, which we are not currently doing.
    # if cfg.mup:
    #     cfg.attn_cfg["softmax_scale"] = cfg.head_dim

    filtered_kwargs = {
        k: v for k, v in asdict(cfg).items() if k in signature(MambaConfig).parameters
    }
    # Force the model to be transformer-only:
    filtered_kwargs["attn_layer_idx"] = list(range(cfg.n_layer))

    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = MambaLMHeadModel(MambaConfig(**filtered_kwargs), device=device)
    if cfg.mup:
        # apply_mup_init calls into a very similar init fuction as the MambaLMHeadModel constructor
        # essentially, just possibly rescaling some of the generated weights. So, reset the seed
        # again to made the d_model == base_d_model mup and non cases coincide.
        torch.cuda.manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        print("Applying mup param init")
        apply_mup_init(model, cfg)
    return model
