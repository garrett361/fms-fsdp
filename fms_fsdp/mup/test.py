from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from fms_fsdp.mup import apply_mup_init, get_mup_optim_iter, mup_config, get_transformer

import torch

num_heads = 2
n_layer = 2
head_dim = 128
d_model = 2 * head_dim
seq_length = 64
device = "cuda" if torch.cuda.is_available() else "cpu"


def test_cfg():
    cfg = mup_config(
        d_model=d_model, head_dim=head_dim, n_layer=n_layer, seq_length=seq_length
    )
    assert cfg


def test_get_transformer():
    cfg = mup_config(
        d_model=d_model, head_dim=head_dim, n_layer=n_layer, seq_length=seq_length
    )
    model = get_transformer(cfg)
    inputs = torch.randint(cfg.vocab_size, size=(1, cfg.seq_length), device=device)
    outputs = model(inputs)


def test_mup_init():
    cfg = mup_config(
        d_model=d_model,
        head_dim=head_dim,
        n_layer=n_layer,
        seq_length=seq_length,
        mup=True,
        mup_base_d_model=d_model // 2,
    )
    model = get_transformer(cfg)
    apply_mup_init(model, cfg)


class TestMupOptim:
    def test_mup_optim_iter(self):
        cfg = mup_config(
            d_model=d_model,
            head_dim=head_dim,
            n_layer=n_layer,
            seq_length=seq_length,
            mup=True,
            mup_base_d_model=d_model // 2,
        )
        model = get_transformer(cfg)
        get_mup_optim_iter(model, cfg)

    def test_width_equals_base_limit(self):
        cfg = mup_config(
            d_model=d_model,
            head_dim=head_dim,
            n_layer=n_layer,
            seq_length=seq_length,
            mup=True,
            mup_base_d_model=d_model,
        )
        model = get_transformer(cfg)
        optim_iter = get_mup_optim_iter(model, cfg)
        for p_lr_dict in optim_iter:
            assert p_lr_dict["lr"] == cfg.learning_rate


def test_coord_check():
    transformer_only_config = MambaConfig(
        d_model=d_model,
        d_intermediate=d_intermediate,
        n_layer=n_layer,
        attn_layer_idx=list(range(n_layer)),
        vocab_size=2048,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )
    model = MambaLMHeadModel(transformer_only_config, device=device)
    inputs = torch.randint(
        transformer_only_config.vocab_size, size=(1, seq_length), device=device
    )
    from coord_check import get_stats

    results_list = []
    get_stats(model, inputs, results_list)
    print(results_list)


class TestMupOptimTransOnly:
    num_heads = 2
    n_layer = 2
    head_dim = 128
    seq_len = 64
    d_model = num_heads * head_dim
    d_intermediate = 4 * d_model
    out_proj_bias = qkv_proj_bias = False
    attn_cfg = {
        "causal": True,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "out_proj_bias": out_proj_bias,
        "qkv_proj_bias": qkv_proj_bias,
        "rotary_emb_dim": 64,
        "softmax_scale": head_dim,
    }
    lr = 1.0
    tie_embeddings = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MambaConfig(
        d_model=d_model,
        d_intermediate=d_intermediate,
        n_layer=n_layer,
        attn_layer_idx=list(range(n_layer)),  # Transformer blocks only
        vocab_size=2048,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )

    def test_width_equals_base_limit(self):
        # All lrs should be the same in the `width == base_width` limit
        model = MambaLMHeadModel(self.cfg, device=self.device)
        optim_iter = mup_mamba.get_mup_optim_iter(model, self.cfg)
        for p_lr_dict in optim_iter:
            assert p_lr_dict["lr"] == self.lr
