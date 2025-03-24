import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from fms_fsdp.pp._mamba import MambaLMHeadModelPP


class TestModel:
    d_model = 256
    d_state = 128
    ngroups = 1
    expand = 2
    d_conv = 4
    d_inner = expand * d_model
    d_ssm = d_inner
    embed_dim = d_model
    head_dim = 64
    num_heads = d_model // head_dim
    ssm_cfg = {"layer": "Mamba2"}
    vocab_size = 1024
    n_layer = 2
    attn_layer_idx = [n_layer - 1]
    attn_cfg = {
        "causal": True,
        "d_conv": 0,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": head_dim // 2,
    }
    cfg = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=attn_layer_idx,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )

    batch_size = 2
    seqlen = 64
    device = "cuda"
    dtype = torch.bfloat16
    factory_kwargs = {"device": "cuda", "dtype": dtype}

    def get_model(self) -> MambaLMHeadModel:
        return MambaLMHeadModel(self.cfg, **self.factory_kwargs)

    def get_model_pp(self) -> MambaLMHeadModelPP:
        return MambaLMHeadModelPP(self.cfg, **self.factory_kwargs)

    def test_model_equality(self) -> None:
        """
        Verify
        """
        torch.manual_seed(42)
        model = self.get_model()
        torch.manual_seed(42)
        model_pp = self.get_model_pp()

        inputs = torch.randint(
            self.vocab_size, size=(self.batch_size, self.seqlen), device=self.device
        )
        outputs = model(inputs).logits
        outputs_pp = model_pp(inputs)
        torch.testing.assert_close(outputs, outputs_pp)
