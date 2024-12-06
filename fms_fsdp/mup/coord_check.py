from mamba_ssm.modules.block import Block
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from typing import Any, Optional, Union
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

ALL_STATS = ("l2_mean", "std", "mean", "var", "l1_mean")


class StatsHook:
    def __init__(
        self,
        module: nn.Module,
        name: str,
        results_list: list[dict],
        d_model: int,
        other_data: Optional[dict] = None,
    ) -> None:
        self.module = module
        self.name = name
        self.d_model = d_model
        self.results_list = results_list
        self._hook = module.register_forward_hook(self)
        self._step = 0
        self.other_data = other_data or {}

    def __call__(self, module: nn.Module, args: Any, output: Any) -> None:
        results = {"name": self.name, "d_model": self.d_model, "step": self._step}
        results = {**results, **self.other_data}
        with torch.no_grad():
            # Grab the hidden states of the block tuple
            if isinstance(module, Block):
                output = output[0]
            results["mean"] = output.mean().item()
            results["l1_mean"] = output.abs().mean().item()
            results["l2_mean"] = output.pow(2).mean().item()
            results["std"] = output.std().item()
            results["var"] = output.var().item()
        self.results_list.append(results)
        self._step += 1

    def remove(self) -> None:
        self._hook.remove()


def get_stats(
    model: MambaLMHeadModel,
    optimizer: torch.optim.Optimizer,
    train_steps: int,
    d_model: int,
    seq_len: int,
    seed: int,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    results_list: list[dict],
) -> None:
    hooks = []
    embedding = model.backbone.embedding
    other_data = {"seed": seed, "seq_len": seq_len}
    hooks.append(
        StatsHook(
            embedding,
            "embedding",
            results_list=results_list,
            d_model=d_model,
            other_data=other_data,
        )
    )

    blocks = model.backbone.layers
    for idx, block in enumerate(blocks):
        hooks.append(
            StatsHook(
                block,
                f"block_{idx}",
                results_list=results_list,
                d_model=d_model,
                other_data=other_data,
            )
        )

    lm_head = model.lm_head
    hooks.append(
        StatsHook(
            lm_head,
            "lm_head",
            results_list=results_list,
            d_model=d_model,
            other_data=other_data,
        )
    )

    model.train()
    for step in tqdm(range(train_steps), desc="step"):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            outputs = outputs.logits if hasattr(outputs, "logits") else outputs
            ce_loss = torch.nn.CrossEntropyLoss()
            loss = ce_loss(outputs.view(-1, outputs.size(-1)), labels.view(-1).long())
        loss.backward()
        optimizer.step()

    for hook in hooks:
        hook.remove()


# TODO: @goon - Title on plot with lr, other data
def plot_from_df(
    df: pd.DataFrame,
    y: str,
    save_path: Optional[Union[pathlib.Path, str]] = None,
    ncols: int = 4,
    title: Optional[str] = None,
) -> matplotlib.figure.Figure:
    if y not in ALL_STATS:
        raise ValueError(f"{y=} must be in {ALL_STATS}")
    nrows = (len(df.step.unique()) + ncols - 1) // ncols
    fig, axs = plt.subplots(
        ncols=ncols, nrows=nrows, sharey=True, figsize=(4 * ncols, 4 * nrows)
    )
    for step in df.step.unique():
        row, col = divmod(step, ncols)
        plot = sns.lineplot(
            data=df[df.step == step], x="d_model", y=y, hue="name", ax=axs[row, col]
        )
        # Log-log for positive quantities:
        plot.set(xscale="log")
        if y != "mean":
            plot.set(yscale="log")
        plot.get_legend().remove()
        axs[row, col].set_title(f"Step {step.item()}")

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.2, 0.5))

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    if save_path:
        fig.savefig(save_path, dpi=256, bbox_inches="tight")

    return fig
