from mamba_ssm.modules.block import Block
import torch.nn as nn
import torch
from typing import Any

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# TODO: @goon - LM Head layer
_DEFAULT_CLASSES = [nn.Embedding, MambaLMHeadModel, Block]


class StatsHook:
    def __init__(
        self, module: nn.Module, name: str, results_list: list[dict], width: int
    ) -> None:
        self.module = module
        self.name = name
        self.width = width
        self.results_list = results_list
        self._hook = module.register_forward_hook(self)

    def __call__(self, module: nn.Module, args: Any, output: Any) -> None:
        results = {"name": self.name, "width": self.width}
        with torch.no_grad():
            # Grab the hidden states of the block tuple
            if isinstance(module, Block):
                output = output[0]
            results["mean"] = output.mean().item()
            results["l2_mean"] = output.pow(2).mean().item()
            results["std"] = output.std().item()
            results["var"] = output.var().item()
        self.results_list.append(results)

    def remove(self) -> None:
        self._hook.remove()


def get_stats(
    model: MambaLMHeadModel, width: int, inputs: torch.Tensor, results_list: list[dict]
) -> None:
    hooks = []
    embedding = model.backbone.embedding
    hooks.append(StatsHook(embedding, "embedding", results_list, width))

    blocks = model.backbone.layers
    for idx, block in enumerate(blocks):
        hooks.append(StatsHook(block, f"block_{idx}", results_list, width))

    lm_head = model.lm_head
    hooks.append(StatsHook(lm_head, "lm_head", results_list, width))

    model.eval()
    with torch.no_grad():
        _ = model(inputs)
    model.train()

    for hook in hooks:
        hook.remove()
