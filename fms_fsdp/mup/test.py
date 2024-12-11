from fms_fsdp.mup import get_mup_optim_iter, mup_config, get_transformer, get_optimizer
import pytest

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


class TestInit:
    @pytest.mark.parametrize("mup", (True, False))
    def test_init(self, mup):
        cfg = mup_config(
            d_model=d_model,
            head_dim=head_dim,
            n_layer=n_layer,
            seq_length=seq_length,
            mup=mup,
            mup_base_d_model=d_model // 2,
        )
        model = get_transformer(cfg)

    def test_mup_init_equivalence(self):
        # mup and non-mup init should coincide when `d_model == mup_base_d_model` and
        # mup_simple_scaling_impl=True
        kwargs = dict(
            d_model=d_model, head_dim=head_dim, n_layer=n_layer, seq_length=seq_length
        )
        cfg = mup_config(mup=False, **kwargs)
        mup_cfg = mup_config(
            mup=True, mup_simple_scaling_impl=True, mup_base_d_model=d_model, **kwargs
        )

        torch.manual_seed(cfg.seed)
        model = get_transformer(cfg)
        torch.manual_seed(cfg.seed)
        mup_model = get_transformer(mup_cfg)

        with torch.no_grad():
            for (p_name, p), (mup_p_name, mup_p) in zip(
                model.named_parameters(), mup_model.named_parameters()
            ):
                assert p_name == mup_p_name
                try:
                    torch.testing.assert_close(p, mup_p)
                except Exception as e:
                    print(f"Failed on parameter {p_name}")
                    raise e


class TestOptim:
    @pytest.mark.parametrize("mup", (True, False))
    @pytest.mark.parametrize("optim", ("adamw", "sgd"))
    def test_get_optimizer(self, optim, mup):
        cfg = mup_config(
            d_model=d_model,
            head_dim=head_dim,
            n_layer=n_layer,
            seq_length=seq_length,
            mup=mup,
            mup_base_d_model=d_model / 100,
            optim=optim,
        )
        model = get_transformer(cfg)
        optimizer = get_optimizer(cfg, model)
        assert optimizer is not None

    @pytest.mark.parametrize("optim", ("adamw", "sgd"))
    def test_mup_optim_iter(self, optim):
        cfg = mup_config(
            d_model=d_model,
            head_dim=head_dim,
            n_layer=n_layer,
            seq_length=seq_length,
            mup=True,
            mup_base_d_model=d_model // 2,
            optim=optim,
        )
        model = get_transformer(cfg)
        get_mup_optim_iter(model, cfg)

    @pytest.mark.parametrize("optim", ("adamw", "sgd"))
    def test_width_equals_base_limit(self, optim):
        cfg = mup_config(
            d_model=d_model,
            head_dim=head_dim,
            n_layer=n_layer,
            seq_length=seq_length,
            mup=True,
            mup_base_d_model=d_model,
            optim=optim,
        )
        model = get_transformer(cfg)
        optim_iter = get_mup_optim_iter(model, cfg)
        for p_lr_dict in optim_iter:
            assert p_lr_dict["lr"] == cfg.learning_rate
