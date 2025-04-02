import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.utils.config_utils import get_model_config


class TestBuildModels:
    def test_mamba_moe_30b(self) -> None:
        config_data = get_model_config("mamba_moe_30b")
        mamba_config = MambaConfig(**config_data)
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 30_072_016_000  # ~ 30B

    def test_mamba_moe_lite(self) -> None:
        config_data = get_model_config("mamba_moe_lite")
        mamba_config = MambaConfig(**config_data)
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 2_516_267_712  # ~ 2B
