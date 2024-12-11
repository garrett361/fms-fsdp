import datetime
import math
import os
import random
import textwrap
import time
from contextlib import nullcontext
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Union

import fire
import torch
import torch.optim as optim
import wandb
from mamba_ssm.modules.block import Block
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp.mup import (
    get_cfg_from_kwargs,
    get_optimizer,
    get_transformer,
    mup_config,
)
from fms_fsdp.utils.dataloader_utils import parse_data_args
from fms_fsdp.utils.dataset_utils import (
    ArrowHandler,
    AutoHandler,
    BufferDataset,
    ParquetHandler,
    PreloadBufferDataset,
    PreprocessDataset,
    SamplingDataset,
    ScalableShardDataset,
    StreamingDocDataset,
)
from fms_fsdp.utils.train_utils import (
    get_policies,
    train,
)


@cache
def get_rank() -> int:
    return int(os.environ["RANK"])


@cache
def get_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


@cache
def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


@cache
def get_device() -> int:
    return torch.cuda.device(get_local_rank())


"""
Minimal multi-gpu, single-node script for quick training with FSDP.  No checkpointing.
"""


def causal_lm(data_seq, prompt_len=1):
    """
    Perform causal language modeling by right-shifting the input sequence.
    Sets first prompt_len tokens to be ignored by the loss.
    """
    data_seq = data_seq.to(torch.int)
    t = data_seq.clone()[1:]
    data_seq = data_seq[:-1]
    t[:prompt_len] = -100
    return data_seq, t


def get_data_loader(cfg, postprocess=[causal_lm]):
    """
    Pytorch dataloader for stateful, distributed, and rescalable causal language model (CLM) training.
    Assumes underlying data is sequences of integer values.
    ...
    Args
    ----
    cfg : dataclass
        Training config containing seq len, dataset, dataset weight, datapath, etc. arguments
    postprocess : List[Callable]
        Any task-specific postprocessing to apply before handing over data. Steps will apply in
        the order provided by the user. For CLM training, use postprocess=[causal_lm].
    """
    _handler_map = {
        "arrow": ArrowHandler,
        "hf_parquet": ParquetHandler,
        "auto": AutoHandler,
    }

    datasets, weights = parse_data_args(cfg.datasets, cfg.weights)

    # Base streaming dataset. Returns doc chunks in sequence.
    # Implements dataset sampling and rescalability.
    droplist = [
        int(x.strip()) for x in cfg.strip_tokens.split(",") if len(x.strip()) > 0
    ]
    droplist = droplist + [cfg.bos_token, cfg.eos_token, cfg.bol_token, cfg.eol_token]
    assert (
        cfg.file_type in _handler_map
    ), f"File type {cfg.file_type} is not recognized ({list(_handler_map.keys())})"
    if cfg.file_type == "hf_parquet" or cfg.file_type == "auto":
        filehandler = _handler_map[cfg.file_type](cfg.tokenizer_path, cfg.col_name)
    else:
        filehandler = _handler_map[cfg.file_type]
    # Base reader layer
    data = StreamingDocDataset(
        cfg.data_path,
        get_rank(),
        cfg.world_size,
        filehandler,
        cfg.eos_token,
        bos_token=cfg.bos_token,
        strip_tokens=set(droplist),
        min_length=3,
        seed=cfg.seed,
    )
    # Add rescaling/resharding
    data = ScalableShardDataset(
        data,
        cfg.eos_token,
        n_logical_shards=cfg.logical_shards,
    )
    # Add multi-dataset handling
    data = SamplingDataset(
        cfg.data_path,
        data,
        cfg.eos_token,
        datasets=datasets,
        weights=weights,
        verbose=(get_rank() == 0),
    )
    # Wrap above dataset in packing logic to form constant-length lines.
    data = BufferDataset(
        data,
        cfg.seq_length if causal_lm not in postprocess else cfg.seq_length + 1,
        bos_token=cfg.bol_token,
        eos_token=cfg.eol_token,
        pack_hard=True,
    )
    # Shuffle outputs in length 10k buffer. Consecutive lines appear 10k steps apart on average.
    data = PreloadBufferDataset(data, 10000)

    # Apply desired postprocessing steps in sequence
    data = PreprocessDataset(data, torch.IntTensor)
    for p in postprocess:
        data = PreprocessDataset(data, p)

    return torch.utils.data.DataLoader(
        data, num_workers=cfg.num_workers, batch_size=cfg.batch_size
    )


def get_dummy_loader(cfg):
    """
    A simple dummy dataloader yielding incrementing vocab indices in an infinite loop
    """

    class SteadyCounter(torch.utils.data.IterableDataset):
        # Spit out incremental counts of constant length l, modulo vocab size v
        def __init__(self, l, v):
            self.i = 0
            self.l = l
            self.v = v

        def __iter__(self):
            while True:
                out = torch.IntTensor(
                    [x % self.v for x in range(self.i, self.i + self.l)]
                )
                yield out, out
                self.i += self.l

    data = SteadyCounter(cfg.seq_length, cfg.vocab_size)
    return torch.utils.data.DataLoader(data, batch_size=cfg.batch_size)


def train(
    cfg,
    model,
    train_loader,
    optimizer,
    scheduler,
):
    wrapper = textwrap.TextWrapper(initial_indent=f"[rank={get_rank()}] ")

    def print_rank(s, *args, **kwargs):
        print("\n".join(wrapper.wrap(s)), *args, **kwargs)

    def print_rank0_only(s, *args, **kwargs):
        if not get_rank():
            print_rank(s, *args, **kwargs)

    print_rank0_only(f"Training for {cfg.num_steps} steps")
    print_rank0_only(
        f"\n--> model has {sum(p.numel() for p in model.parameters() if p.requires_grad):.2e} Million params\n"
    )
    print_rank0_only(f"{model=}")
    model.train()

    start = time.time()
    loop_start = time.time()
    loss_history = []
    gnorm_history = []

    total_tokens = (
        cfg.num_steps * cfg.batch_size * cfg.seq_length * cfg.acc_steps * cfg.world_size
    )

    # Each batch is comprised of acc_steps mini batches. acc_steps backwards per optim step.
    for mini_batch_idx, (input, label) in enumerate(train_loader):
        batch_idx, acc_step_idx = divmod(mini_batch_idx + cfg.acc_steps, cfg.acc_steps)
        is_last_mini_batch = acc_step_idx == cfg.acc_steps - 1

        if batch_idx > cfg.num_steps:
            break

        input = input.to(get_device())
        label = label.to(get_device())

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input)
            output = output.logits if hasattr(output, "logits") else output
            ce_loss = torch.nn.CrossEntropyLoss()
            loss = ce_loss(output.view(-1, output.size(-1)), label.view(-1).long())
            loss_history.append(loss.detach().clone())

        (loss / cfg.acc_steps).backward()

        if is_last_mini_batch:
            gnorm = model.clip_grad_norm_(cfg.grad_clip_thresh).item()
            gnorm_history.append(gnorm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx % cfg.report_interval == 0) and is_last_mini_batch:
            mean_loss = torch.stack(loss_history).mean()
            loss_history.clear()

            mean_gnorm = torch.stack(gnorm_history).mean()
            gnorm_history.clear()

            dist.reduce(mean_loss, op=dist.ReduceOp.AVG, dst=0)
            dist.reduce(mean_gnorm, op=dist.ReduceOp.AVG, dst=0)

            dist.barrier()
            if not get_rank():
                current_loss = mean_loss.item()
                current_gnorm = mean_gnorm.item()

                elapsed_time = time.time() - loop_start
                current_lr = scheduler.get_last_lr()[0]
                tokens_seen = (
                    batch_idx
                    * cfg.world_size
                    * cfg.batch_size
                    * cfg.seq_length
                    * cfg.acc_steps
                )
                current_step_time = (time.time() - start) / cfg.report_interval
                overall_step_time = elapsed_time / batch_idx
                current_throughput = int(
                    cfg.world_size
                    * cfg.acc_steps
                    * cfg.batch_size
                    * cfg.seq_length
                    / current_step_time
                )
                overall_throughput = int(
                    cfg.world_size
                    * cfg.acc_steps
                    * cfg.batch_size
                    * cfg.seq_length
                    / overall_step_time
                )
                reserved_mem = torch.cuda.max_memory_reserved(
                    device=torch.cuda.current_device()
                )
                allocated_mem = torch.cuda.max_memory_allocated(
                    device=torch.cuda.current_device()
                )
                tokens_remaining = total_tokens - tokens_seen
                secs_remaining = tokens_remaining / current_throughput

                print_rank0_only("step:", batch_idx)
                print_rank0_only("loss:", current_loss)
                print_rank0_only("LR:", current_lr)
                print_rank0_only("tokens seen:", tokens_seen)
                print_rank0_only("gradient norm:", current_gnorm)
                print_rank0_only("reserved memory:", reserved_mem)
                print_rank0_only("allocated memory:", allocated_mem)
                print_rank0_only("current step time:", current_step_time)
                print_rank0_only("overall step time:", overall_step_time)
                print_rank0_only("current token per gpu per sec:", current_throughput)
                print_rank0_only("overall token per gpu per sec:", overall_throughput)
                print_rank0_only(
                    "overall token per day:",
                    int(tokens_seen / elapsed_time * 3600 * 24),
                )
                print_rank0_only(
                    "approx time remaining (H:M:S):",
                    str(datetime.timedelta(seconds=secs_remaining)),
                    "\n",
                )
                if cfg.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "gradient norm": current_gnorm,
                        "token seen": tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
                    if cfg.tracker == "wandb":
                        tracker_fn = wandb.log
                    tracker_fn(vals_to_track, step=batch_idx)

            start = time.time()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

    return mean_loss


def get_scheduler(cfg: mup_config, optimizer: optim.Optimizer) -> LambdaLR:
    # LR schedule  (cosine 0.01 decay)
    warmup_interval = min(1000, cfg.num_steps // 10)
    schedule = lambda x: min(
        1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,
        0.01
        + 0.5
        * (1 - 0.01)
        * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
    )

    scheduler = LambdaLR(optimizer, lambda x: schedule(x))

    return scheduler


def main(cfg: mup_config) -> None:
    def setup(cfg, rank) -> None:
        torch.cuda.manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        os.environ["RANK"] = os.environ["LOCAL_RANK"] = str(rank)  # single-node
        os.environ["WORLD_SIZE"] = str(cfg.world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        os.environ["TRITON_CACHE_DIR"] = os.path.join(
            Path.home(), ".triton", "cache", str(rank)
        )
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)

    def target(cfg: mup_config, rank: int) -> None:
        setup(cfg, rank)
        try:
            dist.init_process_group(backend="nccl")

            # Non-FSDP model
            model = get_transformer(cfg)
            block = Block
            (
                mixed_precision_policy,
                wrapping_policy,
                sharding_strategy_policy,
                apply_selective_ac,
                param_init_fn,
            ) = get_policies(cfg, rank, block)
            model = FSDP(
                model,
                auto_wrap_policy=wrapping_policy,
                mixed_precision=mixed_precision_policy,
                sharding_strategy=sharding_strategy_policy,
                use_orig_params=cfg.use_torch_compile,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                param_init_fn=param_init_fn,
            )

            # fsdp activation checkpointing
            if cfg.fsdp_activation_checkpointing:
                if rank == 0:
                    print("--> applying FSDP activation checkpointing...")
                apply_selective_ac(model, p=cfg.selective_checkpointing)

            # torch compile
            if cfg.use_torch_compile:
                if rank == 0:
                    print("--> enabling torch compile...")
                # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
                torch._dynamo.config.accumulated_cache_size_limit = 128
                model = torch.compile(model)

            optimizer = get_optimizer(cfg, model)
            scheduler = get_scheduler(cfg, model)

            # get data loader
            print("Constructing datasets...")
            if not cfg.use_dummy_dataset:
                train_loader = get_data_loader(cfg)
            else:
                train_loader = get_dummy_loader(cfg)
            print("Datasets constructed!")

            # Train
            print(f"Training for {cfg.num_steps} steps")
            train(
                cfg,
                model,
                train_loader,
                optimizer,
                scheduler,
            )
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":

    def run(**kwargs) -> None:
        cfg = get_cfg_from_kwargs(**kwargs)
        ctx: Union[wandb.Run, nullcontext]
        if cfg.tracker == "wandb":
            ctx = wandb.init(
                project=cfg.tracker_project_name,
                dir=cfg.tracker_dir,
                resume="never",
                id=None,
                config=asdict(cfg),
            )
        else:
            ctx = nullcontext()
        with ctx as run:
            # Important: for some reason there are frequent hangs if we use a non-trivial id in
            # wandb.init when this script is run under mutiprocessing, but it works fine if we
            # just set the name by hand.
            if cfg.tracker == "wandb":
                run.name = cfg.tracker_run_id
            main(cfg)

    fire.Fire(run)
