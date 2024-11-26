from mamba_ssm.modules.block import Block
from tqdm import tqdm
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
        self._step = 0

    def __call__(self, module: nn.Module, args: Any, output: Any) -> None:
        results = {"name": self.name, "width": self.width, "step": self._step}
        with torch.no_grad():
            # Grab the hidden states of the block tuple
            if isinstance(module, Block):
                output = output[0]
            results["mean"] = output.mean().item()
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
    width: int,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    results_list: list[dict],
) -> None:
    hooks = []
    embedding = model.backbone.embedding
    hooks.append(StatsHook(embedding, "embedding", results_list, width))

    blocks = model.backbone.layers
    for idx, block in enumerate(blocks):
        hooks.append(StatsHook(block, f"block_{idx}", results_list, width))

    lm_head = model.lm_head
    hooks.append(StatsHook(lm_head, "lm_head", results_list, width))

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
