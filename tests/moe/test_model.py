import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.utils.config_utils import get_model_config


class TestBuildModels:
    def test_mamba_moe_lite(self) -> None:
        mamba_config = get_model_config("mamba_moe_lite")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 2_516_267_712  # ~ 2B

    def test_mamba_30b_moe(self) -> None:
        mamba_config = get_model_config("mamba_30b_moe")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 27_911_687_584  # ~ 27B

    def test_mamba_120b_moe(self) -> None:
        mamba_config = get_model_config("mamba_120b_moe")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 117_685_942_784  # ~ 117B

    def test_mamba_236b_moe(self) -> None:
        mamba_config = get_model_config("mamba_236b_moe")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 236_844_910_400  # ~ 236B

    def test_mamba_105b_moe_sparse(self) -> None:
        mamba_config = get_model_config("mamba_105b_moe_sparse")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 104_032_013_728  # ~ 104B

    def test_mamba_dev_cfg(self) -> None:
        n_layers = 3
        n_routed_experts = 7
        n_activated_experts = 2
        mamba_config = get_model_config(
            f"mamba_moe_dev_{n_layers}_layer_{n_routed_experts}_exp_{n_activated_experts}_act"
        )
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        assert len(model.backbone.layers) == n_layers
        first_moe = model.backbone.layers["0"].mlp
        assert first_moe.n_routed_experts == n_routed_experts
        assert first_moe.n_activated_experts == n_activated_experts

    def test_ds_v2_lite(self) -> None:
        mamba_config = get_model_config("deepseek-v2-lite")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 15_788_259_328  # ~ 16B

    def test_ds_v2(self) -> None:
        mamba_config = get_model_config("deepseek-v2")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 246_920_442_880  # ~ 246B (NOTE: @goon - should be 236B per HF)

    def test_ds_v3(self) -> None:
        mamba_config = get_model_config("deepseek-v3")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        params = sum(p.numel() for p in model.parameters())
        assert params == 688_268_357_120  # ~688 B (NOTE: @goon - should be 671B per HF)
