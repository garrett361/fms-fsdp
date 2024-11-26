from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import mup_mamba

import torch

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
tie_embeddings = False
device = "cuda" if torch.cuda.is_available() else "cpu"


def test_mamba_init():
    mamba_config = MambaConfig(
        d_model=512,
        d_intermediate=0,
        n_layer=1,
        vocab_size=2048,
    )
    model = MambaLMHeadModel(mamba_config)
    print(model)


def test_transformer_only():
    config = MambaConfig(
        d_model=d_model,
        d_intermediate=d_intermediate,
        n_layer=n_layer,
        attn_layer_idx=list(range(n_layer)),  # Transformer blocks only
        vocab_size=2048,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )
    model = MambaLMHeadModel(config, device=device)
    inputs = torch.randint(config.vocab_size, size=(1, seq_len), device=device)
    print(model)
    outputs = model(inputs)
    print(outputs)


def test_mup():
    mamba_config = MambaConfig(
        d_model=d_model,
        d_intermediate=d_intermediate,
        n_layer=n_layer,
        attn_layer_idx=list(range(n_layer)),
        vocab_size=2048,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )
    model = MambaLMHeadModel(mamba_config)
    mup_mamba.apply_mup_init(model)
    print("done")


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
        transformer_only_config.vocab_size, size=(1, seq_len), device=device
    )
    from coord_check import get_stats

    results_list = []
    get_stats(model, inputs, results_list)
    print(results_list)
