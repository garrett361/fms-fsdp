import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.moe_utils import get_total_exp_and_active_params
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

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
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 27_911_687_584
        assert exp == 25_367_150_592
        assert active == 5_715_430_816

    def test_mamba_120b_moe(self) -> None:
        mamba_config = get_model_config("mamba_120b_moe")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 117_685_942_784
        assert exp == 112_742_891_520
        assert active == 11_989_481_984

    def test_mamba_236b_moe(self) -> None:
        mamba_config = get_model_config("mamba_236b_moe")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 236_844_910_400
        assert exp == 226_492_416_000
        assert active == 21_677_115_200

    def test_mamba_105b_moe_sparse(self) -> None:
        mamba_config = get_model_config("mamba_105b_moe_sparse")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 104_032_013_728
        assert exp == 101_468_602_368
        assert active == 5_734_305_184

    def test_ds_v2_lite(self) -> None:
        mamba_config = get_model_config("deepseek-v2-lite")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 15_788_259_328  # (NOTE: @goon - should be 16B per HF)
        assert exp == 14_394_851_328
        assert active == 2_742_925_312  # (NOTE: @goon - should be 2.4B per HF)

    def test_ds_v2(self) -> None:
        mamba_config = get_model_config("deepseek-v2")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 246_920_442_880  # ~ 246B (NOTE: @goon - should be 236B per HF)
        assert exp == 222_717_542_400
        assert active == 32_554_808_320  # ~15B (NOTE: @goon - should be 21B per HF)

    def test_ds_v3(self) -> None:
        mamba_config = get_model_config("deepseek-v3")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 688_268_357_120  # ~688 B (NOTE: @goon - should be 671B per HF)
        assert exp == 653_908_770_816
        assert active == 54_794_235_392  # ~52B (NOTE: @goon - should be ~37B?)

    def test_llama4_maverick(self) -> None:
        mamba_config = get_model_config("llama4-maverick")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 398_195_266_560  # ~400B
        assert exp == 386_547_056_640
        assert active == 14_668_108_800  # ~15B (NOTE: @goon - should be ~17B per HF)

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

    def test_ds_v2_lite_dev_cfg(self) -> None:
        n_layers = 3
        mamba_config = get_model_config(f"deepseek-v2-lite-dev_{n_layers}_layer")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        assert len(model.backbone.layers) == n_layers

    def test_ds_v2_dev_cfg(self) -> None:
        n_layers = 3
        mamba_config = get_model_config(f"deepseek-v2-dev_{n_layers}_layer")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        assert len(model.backbone.layers) == n_layers

    def test_ds_v3_dev_cfg(self) -> None:
        n_layers = 3
        mamba_config = get_model_config(f"deepseek-v3-dev_{n_layers}_layer")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        assert len(model.backbone.layers) == n_layers
        total, exp, active = get_total_exp_and_active_params(model)

    def test_llama4_maverick_dev_cfg(self) -> None:
        n_layers = 3
        mamba_config = get_model_config(f"llama4-maverick-dev_{n_layers}_layer")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        assert len(model.backbone.layers) == n_layers

    def test_olmoe_7b(self) -> None:
        mamba_config = get_model_config("olmoe-7b")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 7_238_387_712
        assert exp == 6_442_450_944
        assert active == 1_601_243_136

    def test_hybrid_1b(self) -> None:
        mamba_config = get_model_config("hybrid-1b")
        with torch.device("meta"):
            model = MambaLMHeadModel(mamba_config)
        total, exp, active = get_total_exp_and_active_params(model)
        assert total == 1_158_122_960  # ~ 1.1B
        assert exp == 754_974_720
        assert active == 497_520_080


class TestHF:
    def test_granite_4_tiny_preview(self) -> None:
        model_path = "ibm-granite/granite-4.0-tiny-preview"
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        conv = [
            {
                "role": "user",
                "content": "You have 10 liters of a 30% acid solution. How many liters of a 70% acid solution must be added to achieve a 50% acid mixture?",
            }
        ]

        input_ids = tokenizer.apply_chat_template(
            conv,
            return_tensors="pt",
            thinking=True,
            return_dict=True,
            add_generation_prompt=True,
        ).to("cuda")

        set_seed(42)
        output = hf_model.generate(
            **input_ids,
            max_new_tokens=42,
        )

        prediction = tokenizer.decode(
            output[0, input_ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(prediction)

        hf_config = AutoConfig.from_pretrained(model_path)

    def test_granite_4_tiny_preview_cfg_conversion(self) -> None:
        model_path = "ibm-granite/granite-4.0-tiny-preview"
        hf_config = AutoConfig.from_pretrained(model_path)
        print("hf_config")
