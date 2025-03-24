from copy import deepcopy

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

    def get_input_toks(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randint(
            self.vocab_size, size=(self.batch_size, self.seqlen), device=self.device
        )

    def get_inputs(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(
            self.batch_size, self.seqlen, self.d_model, **self.factory_kwargs
        )

    def get_model(self) -> MambaLMHeadModel:
        return MambaLMHeadModel(self.cfg, **self.factory_kwargs)

    def get_model_pp(self) -> MambaLMHeadModelPP:
        return MambaLMHeadModelPP(self.cfg, **self.factory_kwargs)

    def test_model_equality(self) -> None:
        """
        Verify our PP class agrees with the standard mamba model.
        """
        torch.manual_seed(42)
        model = self.get_model()
        torch.manual_seed(42)
        model_pp = self.get_model_pp()

        # Test weight init
        torch.testing.assert_close(
            model.backbone.embedding.weight, model_pp.backbone.embedding.weight
        )
        for layer_idx in range(len(model.backbone.layers)):
            for p1, p2 in zip(
                model.backbone.layers[layer_idx].parameters(),
                model_pp.backbone.layers[str(layer_idx)].parameters(),
            ):
                torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(model.lm_head.weight, model_pp.lm_head.weight)

        # And output equality
        inputs = self.get_input_toks()
        outputs = model(inputs).logits
        outputs_pp, _ = model_pp(inputs)
        torch.testing.assert_close(outputs, outputs_pp)

    def test_first_stage_pp(self) -> None:
        """
        Verify we can remove every stage but the embedding layer.
        """
        model_pp = self.get_model_pp()
        model_pp.lm_head = None
        for layer_idx in model_pp.backbone.layers:
            model_pp.backbone.layers[layer_idx] = None

        inputs = self.get_input_toks()
        outputs_pp, residuals = model_pp(inputs)
        assert isinstance(outputs_pp, torch.Tensor)
        assert residuals is None

    def test_middle_stage_pp(self) -> None:
        """
        Verify we can remove the embedding and lm head.
        """
        model_pp = self.get_model_pp()
        model_pp.lm_head = model_pp.backbone.embedding = None

        residuals = inputs = self.get_inputs()
        outputs_pp, residuals = model_pp(inputs, residuals)
        assert isinstance(outputs_pp, torch.Tensor)
        assert isinstance(residuals, torch.Tensor)

    def test_final_stage_pp(self) -> None:
        """
        Verify we can remove every stage but the lm head layer.
        """
        model_pp = self.get_model_pp()
        model_pp.backbone.embedding = None
        for layer_idx in model_pp.backbone.layers:
            model_pp.backbone.layers[layer_idx] = None

        residuals = inputs = self.get_inputs()
        outputs_pp, residuals = model_pp(inputs, residuals)
        assert isinstance(outputs_pp, torch.Tensor)
        assert residuals is None

    def test_fake_pp(self) -> None:
        """
        Verify a fake mock pipeline works.
        """
        torch.manual_seed(42)
        model = self.get_model_pp()

        model_pp_first = deepcopy(model)
        model_pp_middle = deepcopy(model)
        model_pp_last = deepcopy(model)

        model_pp_first.lm_head = None
        for layer_idx in model_pp_first.backbone.layers:
            model_pp_first.backbone.layers[layer_idx] = None

        model_pp_middle.lm_head = model_pp_middle.backbone.embedding = None

        model_pp_last.backbone.embedding = None
        for layer_idx in model_pp_last.backbone.layers:
            model_pp_last.backbone.layers[layer_idx] = None

        inputs = self.get_input_toks()
        outputs, _ = model(inputs)

        outputs_pp_first, residuals = model_pp_first(inputs)
        outputs_pp_middle, residuals = model_pp_middle(outputs_pp_first, residuals)
        outputs_pp_last, _ = model_pp_last(outputs_pp_middle, residuals)

        torch.testing.assert_close(outputs, outputs_pp_last)
