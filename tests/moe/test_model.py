import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.utils.config_utils import get_model_config


class TestBuildModels:
    def test_mamba_moe_lite(self) -> None:
        config_data = get_model_config("mamba_moe_lite")
        mamba_config = MambaConfig(**config_data)
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 2_516_267_712  # ~ 2B

    def test_mamba_30b_moe(self) -> None:
        config_data = get_model_config("mamba_30b_moe")
        mamba_config = MambaConfig(**config_data)
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 27_911_687_584  # ~ 27B

    def test_mamba_120b_moe(self) -> None:
        config_data = get_model_config("mamba_120b_moe")
        mamba_config = MambaConfig(**config_data)
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 117_685_942_784  # ~ 117B

    def test_mamba_236b_moe(self) -> None:
        config_data = get_model_config("mamba_236b_moe")
        mamba_config = MambaConfig(**config_data)
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 236_844_910_400  # ~ 236B

    def test_mamba_105b_moe_sparse(self) -> None:
        config_data = get_model_config("mamba_105b_moe_sparse")
        mamba_config = MambaConfig(**config_data)
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 104_032_013_728  # ~ 104B
