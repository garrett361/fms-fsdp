import os

import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridAttention,
    GraniteMoeHybridMambaLayer,
    GraniteMoeHybridMLP,
)

from fms_fsdp.utils.checkpointing_utils import (
    get_hf_state_dict_from_ssm_state_dict,
    get_ssm_state_dict_from_hf_model,
)
from fms_fsdp.utils.config_utils import get_model_config

# User must set this
MODEL_DIR = os.environ["HF_TEST_MODEL_DIR"]


def test_convert_and_load_hf() -> None:
    with torch.no_grad():
        # Check that we can load the cfg and tokenizer
        hf_config = AutoConfig.from_pretrained(MODEL_DIR)
        if hf_config.embedding_multiplier != 1.0:
            raise NotImplementedError
        if hf_config.residual_multiplier != 1.0:
            raise NotImplementedError
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        inputs = tokenizer.encode(
            "Why did the chicken cross the road?", return_tensors="pt"
        ).cuda()

        # Load and convert the model from HF to ssm:
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, device_map="cuda", torch_dtype=torch.torch.bfloat16
        )
        hf_logits = hf_model(inputs).logits

        ssm_state_dict = get_ssm_state_dict_from_hf_model(hf_model)

        # Verify we can load into a granite ssm cfg model:
        config_data = get_model_config("granite")
        mamba_config = MambaConfig(**config_data)

        model = MambaLMHeadModel(
            mamba_config, device="cuda", dtype=torch.torch.bfloat16
        )
        model.load_state_dict(ssm_state_dict["model_state"], strict=True)
        assert model.config.attn_cfg["rotary_emb_dim"] == 0, (
            "Model should not have rope configured."
        )

        # # Convert back to hf format and check the state dict agrees. Need to deepcopy, otherwise we
        # # are comparing exactly the same weights against themselves and only the keys are changing.
        # hf_state_dict_again = convert_state_dict_from_mamba_ssm(
        #     deepcopy(model.state_dict()), in_place=False
        # )
        # assert set(hf_state_dict) == set(hf_state_dict_again)
        # for k in hf_state_dict:
        #     torch.testing.assert_close(hf_state_dict[k], hf_state_dict_again[k])
        # del hf_state_dict_again

        logits = model(inputs).logits

        # Check that the top prediction at each position agrees and the average error is small
        torch.testing.assert_close(
            logits.max(dim=-1).indices, hf_logits.max(dim=-1).indices
        )
        tol = 1e-2
        avg_diff = (logits - hf_logits).abs().mean() / logits.abs().mean()
        assert avg_diff < tol, f"{avg_diff=}"

        # Then convert back and reload into the HF model
        hf_model.load_state_dict(
            get_hf_state_dict_from_ssm_state_dict(model.state_dict()), strict=True
        )

        # And check agreement a final time
        hf_logits_again = hf_model(inputs).logits
        torch.testing.assert_close(
            logits.max(dim=-1).indices, hf_logits_again.max(dim=-1).indices
        )
        avg_diff = (logits - hf_logits_again).abs().mean() / logits.abs().mean()
        assert avg_diff < tol, f"{avg_diff=}"

