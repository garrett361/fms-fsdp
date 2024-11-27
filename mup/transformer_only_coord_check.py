from coord_check import get_stats, plot_from_df, ALL_STATS
import torch
from mamba_ssm.models.config_mamba import MambaConfig
import pandas as pd
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from tqdm import tqdm
from pathlib import Path
import argparse
from mup_mamba import apply_mup_init, get_mup_optim_iter


def get_transformer_and_config(
    width: int,
    n_layer: int = 10,
    vocab_size: int = 128256,
    head_dim: int = 128,
    device: str = "cuda",
    mup: bool = False,
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
    }
    if mup:
        attn_cfg["softmax_scale"] = head_dim

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
    if mup:
        print("Applying mup param init")
        apply_mup_init(model)
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
    parser.add_argument("--n_seeds", type=int, default=2)
    parser.add_argument("--mup", action="store_true")
    parser.add_argument("--n_layer", type=int, default=10)
    parser.add_argument("--head_dim", type=int, default=128)
    args = parser.parse_args()

    results_list: list[dict] = []
    # Train repeatedly on fake data
    for seed in tqdm(range(args.seed, args.seed + args.n_seeds), desc="seed"):
        torch.manual_seed(seed)
        inputs_and_labels = torch.randint(
            args.vocab_size, size=(1, args.seq_len + 1), device="cuda"
        )
        inputs = inputs_and_labels[:, :-1]
        labels = inputs_and_labels[:, 1:]
        for width in tqdm(
            range(args.min_width, args.max_width + 1, args.width_step), desc="width"
        ):
            torch.manual_seed(seed)
            model, config = get_transformer_and_config(
                width,
                vocab_size=args.vocab_size,
                mup=args.mup,
                head_dim=args.head_dim,
                n_layer=args.n_layer,
            )
            if args.mup:
                print("Getting mup learning rates")
                optim_args = get_mup_optim_iter(model, args.lr, optim_type="adam")
            else:
                optim_args = model.parameters()
            optimizer = torch.optim.AdamW(
                optim_args, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
            )

            get_stats(
                model=model,
                optimizer=optimizer,
                train_steps=args.train_steps,
                width=width,
                seed=seed,
                seq_len=args.seq_len,
                inputs=inputs,
                labels=labels,
                results_list=results_list,
            )
            del model
        df = pd.DataFrame(results_list)
        parent_dir = Path(__file__).parent.absolute()
        prefix = "trans_coord_check_lr"
        if args.mup:
            prefix += "_mup"
        prefix += f"-{args.lr}lr={args.lr}_seq_len-{args.seq_len}_n_layer-{args.n_layer}_head_dim-{args.head_dim}"
        fig_dir = parent_dir.joinpath("figs/")
        df.to_feather(fig_dir.joinpath(f"{prefix}.feather"))
        title = f"lr={args.lr}, seq_len={args.seq_len}, n_layer={args.n_layer}, head_dim={args.head_dim}"
        for y in ALL_STATS:
            plot_from_df(
                df, y=y, save_path=fig_dir.joinpath(f"{prefix}_{y}.png"), title=title
            )
