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
        attn_layer_idx=list(range(n_layer)),  # Transformer-only blocks
        vocab_size=vocab_size,
        attn_cfg=attn_cfg,
        tie_embeddings=False,
    )
    model = MambaLMHeadModel(config, device=device)
    return model, config


if __name__ == "__main__":
    seq_len = 4096
    train_steps = 3
    vocab_size = 128256
    results_list: list[dict] = []
    inputs_and_labels = torch.randint(vocab_size, size=(1, seq_len + 1), device="cuda")
    inputs = inputs_and_labels[:, :-1]
    labels = inputs_and_labels[:, 1:]
    for width in tqdm(range(512, 4096, 512), desc="width"):
        model, config = get_transformer_and_config(width, vocab_size=vocab_size)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1
        )

        get_stats(
            model=model,
            optimizer=optimizer,
            train_steps=train_steps,
            width=width,
            inputs=inputs,
            labels=labels,
            results_list=results_list,
        )
        del model
    df = pd.DataFrame(results_list)
    parent_dir = Path(__file__).parent.absolute()
    df.to_feather(parent_dir.joinpath("transformer_only_coord_check.feather"))
