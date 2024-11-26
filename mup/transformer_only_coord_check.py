from coord_check import get_stats, plot_from_df
import torch
from mamba_ssm.models.config_mamba import MambaConfig
import pandas as pd
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tqdm import tqdm
from pathlib import Path
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--train_steps", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_width", type=int, default=512)
    parser.add_argument("--max_width", type=int, default=4096)
    parser.add_argument("--width_step", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_list: list[dict] = []
    # Train repeatedly on fake data
    torch.manual_seed(args.seed)
    inputs_and_labels = torch.randint(
        args.vocab_size, size=(1, args.seq_len + 1), device="cuda"
    )
    inputs = inputs_and_labels[:, :-1]
    labels = inputs_and_labels[:, 1:]
    for width in tqdm(
        range(args.min_width, args.max_width + 1, args.width_step), desc="width"
    ):
        torch.manual_seed(args.seed)
        model, config = get_transformer_and_config(width, vocab_size=args.vocab_size)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )

        get_stats(
            model=model,
            optimizer=optimizer,
            train_steps=args.train_steps,
            width=width,
            inputs=inputs,
            labels=labels,
            results_list=results_list,
        )
        del model
    df = pd.DataFrame(results_list)
    parent_dir = Path(__file__).parent.absolute()
    prefix = "transformer_only_coord_check"
    df.to_feather(parent_dir.joinpath(f"{prefix}.feather"))
    plot_from_df(df, save_path=f"{prefix}.png")
