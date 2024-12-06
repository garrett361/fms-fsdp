from fms_fsdp.mup import get_mup_optim_iter, mup_config, get_transformer

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
