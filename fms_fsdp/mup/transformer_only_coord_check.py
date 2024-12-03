from coord_check import get_stats, plot_from_df, ALL_STATS
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
from mup_mamba import get_mup_optim_iter
from fms_fsdp.mup.transformer_only_utils import get_transformer_and_config
from fms_fsdp.mup.mup_mamba import apply_mup_init


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
    parser.add_argument("--use_width_in_mup", action="store_true")
    args = parser.parse_args()

    if args.use_width_in_mup and not args.mup:
        raise ValueError("--use_width_in_mup must be used with --mup")

    results_list: list[dict] = []
    # Train repeatedly on fake data
    widths = list(range(args.min_width, args.max_width + 1, args.width_step))
    for seed in tqdm(range(args.seed, args.seed + args.n_seeds), desc="seed"):
        torch.manual_seed(seed)
        inputs_and_labels = torch.randint(
            args.vocab_size, size=(1, args.seq_len + 1), device="cuda"
        )
        inputs = inputs_and_labels[:, :-1]
        labels = inputs_and_labels[:, 1:]
        for width in tqdm(widths, desc="width"):
            torch.manual_seed(seed)
            model, config = get_transformer_and_config(
                width,
                vocab_size=args.vocab_size,
                mup=args.mup,
                head_dim=args.head_dim,
                n_layer=args.n_layer,
            )
            if args.mup:
                print("Getting mup learning rates and applying init")
                apply_mup_init(model)
                optim_args = get_mup_optim_iter(
                    model=model,
                    lr=args.lr,
                    optim_type="adam",
                    base_width=widths[0],
                    width=width if args.use_width_in_mup else None,
                )
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
        fig_dir = parent_dir.joinpath("figs/")

        prefix = "trans_coord_check"
        if args.mup:
            prefix += "_mup"
            if args.use_width_in_mup:
                prefix += "_with_width"
        prefix += f"_lr-{args.lr}_seq_len-{args.seq_len}_n_layer-{args.n_layer}_head_dim-{args.head_dim}"

        df.to_feather(fig_dir.joinpath(f"{prefix}.feather"))

        title = f"lr={args.lr}, seq_len={args.seq_len}, n_layer={args.n_layer}, head_dim={args.head_dim}, widths={widths}"
        if args.mup:
            title = "(mup) " + title
        for y in ALL_STATS:
            plot_from_df(
                df, y=y, save_path=fig_dir.joinpath(f"{prefix}_{y}.png"), title=title
            )
