import argparse
from pathlib import Path

import pandas as pd
import torch
from mup_mamba import get_mup_optim_iter
from tqdm import tqdm

from coord_check import ALL_STATS, get_stats, plot_from_df
from fms_fsdp.mup import apply_mup_init, get_transformer, mup_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--train_steps", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_d_model", type=int, default=512)
    parser.add_argument("--max_d_model", type=int, default=4096)
    parser.add_argument("--d_model_step", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_seeds", type=int, default=2)
    parser.add_argument("--mup", action="store_true")
    parser.add_argument("--n_layer", type=int, default=10)
    parser.add_argument("--head_dim", type=int, default=128)
    args = parser.parse_args()

    results_list: list[dict] = []
    # Train repeatedly on fake data
    d_models = list(range(args.min_d_model, args.max_d_model + 1, args.d_model_step))
    for seed in tqdm(range(args.seed, args.seed + args.n_seeds), desc="seed"):
        torch.manual_seed(seed)
        inputs_and_labels = torch.randint(
            args.vocab_size, size=(1, args.seq_len + 1), device="cuda"
        )
        inputs = inputs_and_labels[:, :-1]
        labels = inputs_and_labels[:, 1:]
        for d_model in tqdm(d_models, desc="d_model"):
            cfg = mup_config(
                d_model=d_model,
                n_layer=args.n_layer,
                seed=args.seed,
                mup=args.mup,
                head_dim=args.head_dim,
                learning_rate=args.lr,
                vocab_size=args.vocab_size,
            )
            torch.manual_seed(seed)
            model = get_transformer(cfg)
            optim_kwargs = dict(betas=(0.9, 0.95), weight_decay=0.1)
            if args.mup:
                print("Getting mup learning rates and applying init")
                apply_mup_init(model, cfg)
                optim_args = get_mup_optim_iter(model=model, cfg=cfg)

            else:
                optim_args = model.parameters()
                optim_kwargs["lr"] = args.lr
            optimizer = torch.optim.AdamW(optim_args, **optim_kwargs)

            get_stats(
                model=model,
                optimizer=optimizer,
                train_steps=args.train_steps,
                d_model=d_model,
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
        prefix += f"_lr-{args.lr}_seq_len-{args.seq_len}_n_layer-{args.n_layer}_head_dim-{args.head_dim}"

        df.to_feather(fig_dir.joinpath(f"{prefix}.feather"))

        title = f"lr={args.lr}, seq_len={args.seq_len}, n_layer={args.n_layer}, head_dim={args.head_dim}, d_models={d_models}"
        if args.mup:
            title = "(mup) " + title
        for y in ALL_STATS:
            fig_subdir = fig_dir.joinpath(y)
            fig_subdir.mkdir(parents=True, exist_ok=True)
            plot_from_df(
                df, y=y, save_path=fig_subdir.joinpath(f"{prefix}_{y}.png"), title=title
            )
