from coord_check import get_stats
import torch
from mamba_ssm.models.config_mamba import MambaConfig
import pandas as pd
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tqdm import tqdm
from pathlib import Path


def get_transformer_and_config(
    width: int,
    n_layer: int = 10,
    vocab_size: int = 128256,
    head_dim: int = 128,
    device: str = "cuda",
) -> tuple[MambaLMHeadModel, MambaConfig]:
    """
    Get small transformer models. Config based on Cerebras-GPT 111M, roughly.
    """
    num_heads, remainder = divmod(width, head_dim)
    if remainder:
        raise ValueError(f"Choose {width=} divisible by {head_dim=}.")
    attn_cfg = {
        "causal": True,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "out_proj_bias": False,
        "qkv_proj_bias": False,
        "rotary_emb_dim": head_dim // 2,  # Apparently correct for mamba-ssm
        "softmax_scale": head_dim,
    }
    config = MambaConfig(
        d_model=width,
        d_intermediate=4 * width,
        n_layer=n_layer,
        attn_layer_idx=list(range(n_layer)),
        vocab_size=vocab_size,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )
    model = MambaLMHeadModel(config, device=device)
    return model, config


if __name__ == "__main__":
    seq_len = 4096
    vocab_size = 128256
    results_list = []
    inputs = torch.randint(vocab_size, size=(1, seq_len), device="cuda")
    for width in tqdm(range(768, 4096, 256)):
        model, config = get_transformer_and_config(width, vocab_size=vocab_size)
        get_stats(model, width, inputs, results_list)
        del model
    df = pd.DataFrame(results_list)
    parent_dir = Path(__file__).parent.absolute()
    df.to_feather(parent_dir.joinpath("transformer_only_coord_check.feather"))
