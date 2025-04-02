import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.utils.config_utils import get_model_config


def test_build_model():
    config_data = get_model_config("mamba_moe_30b")
    mamba_config = MambaConfig(**config_data)
    with torch.device("meta"):
        model = MambaLMHeadModel(mamba_config)
