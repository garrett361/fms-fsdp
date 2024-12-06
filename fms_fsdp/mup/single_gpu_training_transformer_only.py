import math
import datetime
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Union

import fire
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import LambdaLR

from fms_fsdp.mup.mup_mamba import apply_mup_init, get_mup_optim_iter
from fms_fsdp.mup.transformer_only_utils import get_transformer
from fms_fsdp.utils.dataloader_utils import parse_data_args
from fms_fsdp.utils.dataset_utils import (
    ArrowHandler,
    AutoHandler,
    BufferDataset,
    ParquetHandler,
    PreloadBufferDataset,
    PreprocessDataset,
    SamplingDataset,
    StreamingDocDataset,
)
from fms_fsdp.utils.train_utils import (
    setup_environ_flags,
)
from fms_fsdp.mup import mup_config, get_cfg_from_kwargs


"""
Minimal single-gpu script for quick training.  No checkpointing.
"""


def print_device(*args, **kwargs):
    device = os.environ["CUDA_VISIBLE_DEVICES"]
    if len(device) == 1:
        print(f"[{device=}]: ", *args, **kwargs)
    else:
        print(*args, **kwargs)


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


def get_data_loader(cfg: mup_config, postprocess=[causal_lm]):
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
    rank, world_size = 0, 1
    data = StreamingDocDataset(
        cfg.data_path,
        rank,
        world_size,
        filehandler,
        cfg.eos_token,
        bos_token=cfg.bos_token,
        strip_tokens=set(droplist),
        min_length=3,
        seed=cfg.seed,
    )
    # Add multi-dataset handling
    data = SamplingDataset(
        cfg.data_path,
        data,
        cfg.eos_token,
        datasets=datasets,
        weights=weights,
        verbose=True,
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
    model.train()

    start = time.time()
    loop_start = time.time()
    loss_history = []
    gnorm_history = []

    total_tokens = cfg.num_steps * cfg.batch_size * cfg.seq_length * cfg.acc_steps

    # Each batch is comprised of acc_steps mini batches. acc_steps backwards per optim step.
    for mini_batch_idx, (input, label) in enumerate(train_loader):
        batch_idx, acc_step_idx = divmod(mini_batch_idx + cfg.acc_steps, cfg.acc_steps)
        is_last_mini_batch = acc_step_idx == cfg.acc_steps - 1

        if batch_idx > cfg.num_steps:
            break

        input = input.to("cuda")
        label = label.to("cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input)
            output = output.logits if hasattr(output, "logits") else output
            ce_loss = torch.nn.CrossEntropyLoss()
            loss = ce_loss(output.view(-1, output.size(-1)), label.view(-1).long())
            loss_history.append(loss.detach().clone())

        (loss / cfg.acc_steps).backward()

        if is_last_mini_batch:
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_thresh
            )
            gnorm_history.append(gnorm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx % cfg.report_interval == 0) and is_last_mini_batch:
            mean_loss = torch.stack(loss_history).mean()
            loss_history.clear()

            mean_gnorm = torch.stack(gnorm_history).mean()
            gnorm_history.clear()

            elapsed_time = time.time() - loop_start
            tokens_seen = batch_idx * cfg.batch_size * cfg.seq_length * cfg.acc_steps
            current_loss = mean_loss.item()
            current_lr = scheduler.get_last_lr()[0]
            current_gnorm = mean_gnorm.item()
            current_step_time = (time.time() - start) / cfg.report_interval
            overall_step_time = elapsed_time / batch_idx
            current_throughput = int(
                cfg.acc_steps * cfg.batch_size * cfg.seq_length / current_step_time
            )
            overall_throughput = int(
                cfg.acc_steps * cfg.batch_size * cfg.seq_length / overall_step_time
            )
            reserved_mem = torch.cuda.max_memory_reserved(
                device=torch.cuda.current_device()
            )
            allocated_mem = torch.cuda.max_memory_allocated(
                device=torch.cuda.current_device()
            )
            tokens_remaining = total_tokens - tokens_seen
            secs_remaining = tokens_remaining / current_throughput

            print_device("step:", batch_idx)
            print_device("loss:", current_loss)
            print_device("LR:", current_lr)
            print_device("tokens seen:", tokens_seen)
            print_device("gradient norm:", current_gnorm)
            print_device("reserved memory:", reserved_mem)
            print_device("allocated memory:", allocated_mem)
            print_device("current step time:", current_step_time)
            print_device("overall step time:", overall_step_time)
            print_device("current token per gpu per sec:", current_throughput)
            print_device("overall token per gpu per sec:", overall_throughput)
            print_device(
                "overall token per day:",
                int(tokens_seen / elapsed_time * 3600 * 24),
            )
            print_device(
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


def setup(cfg: mup_config) -> None:
    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    setup_environ_flags()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(Path.home(), ".triton", "cache")


def get_model_optim_scheduler(
    cfg: mup_config,
) -> tuple[nn.Module, optim.Optimizer, LambdaLR]:
    # get model
    # config_data = get_model_config(cfg.model_variant)
    # mamba_config = MambaConfig(**config_data)
    # model = MambaLMHeadModel(mamba_config)
    model, _ = get_transformer(
        width=cfg.d_model,
        n_layer=cfg.n_layer,
        vocab_size=cfg.vocab_size,
        head_dim=cfg.head_dim,
        device="cuda",
        mup=cfg.mup,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_device(f"\n--> model has {total_params / 1e6} Million params\n")
    print_device(f"{model=}")

    # torch compile
    if cfg.use_torch_compile:
        print_device("--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Model init and Optimizer
    if cfg.mup:
        apply_mup_init(model)
        assert cfg.mup_base_d_model is not None  # mypy
        optimizer = optim.AdamW(
            get_mup_optim_iter(model, cfg),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    # Override loaded optim hyperparams with the current values
    for g in optimizer.param_groups:
        g["initial_lr"] = cfg.learning_rate

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

    return model, optimizer, scheduler


def main(cfg: mup_config) -> None:
    model, optimizer, scheduler = get_model_optim_scheduler(cfg)

    # get data loader
    print_device("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg)
    else:
        train_loader = get_dummy_loader(cfg)
    print_device("Datasets constructed!")

    # Train
    print_device(f"Training for {cfg.num_steps} steps")
    train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
    )


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
