import math
import os
from dataclasses import asdict


import time

from typing import Optional
from pathlib import Path

import fire
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from fms_fsdp.utils.dataloader_utils import parse_data_args

from fms_fsdp.utils.config_utils import update_config
from fms_fsdp.utils.train_utils import (
    setup_environ_flags,
)
from dataclasses import dataclass
from fms_fsdp.mup.transformer_only_utils import get_transformer_and_config

from fms_fsdp.mup.mup_mamba import apply_mup_init, get_mup_optim_iter

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


_handler_map = {
    "arrow": ArrowHandler,
    "hf_parquet": ParquetHandler,
    "auto": AutoHandler,
}


"""
Minimal single-gpu script for quick training.  No checkpointing.
"""


@dataclass
class mup_config:
    # model
    width: int = 512
    n_layer: int = 10
    head_dim: int = 128
    mup: bool = False
    ckpt_load_path: str = "/fsx/output/ckpt"
    ckpt_save_path: str = "/fsx/output/ckpt"

    # dataset and dataloader
    use_dummy_dataset: bool = False
    data_path: str = "/fsx/data"
    file_type: str = "arrow"
    col_name: str = "tokens"
    tokenizer_path: str = "/fsx/tokenizer"
    datasets: str = "lang=en/dataset=commoncrawl,lang=en/dataset=webhose,lang=en/dataset=github_clean,lang=de/dataset=wikipedia,lang=es/dataset=wikipedia,lang=fr/dataset=wikipedia,lang=ja/dataset=wikipedia,lang=pt/dataset=wikipedia,lang=en/dataset=wikimedia,lang=en/dataset=uspto,lang=en/dataset=pubmedcentral,lang=en/dataset=arxiv,lang=en/dataset=stackexchange"
    weights: str = "7725,500,550,28,17,22,25,8,100,500,175,250,100"
    seq_length: int = 4096
    vocab_size: int = 128256
    bos_token: Optional[int] = None
    eos_token: int = 0
    bol_token: Optional[int] = None
    eol_token: Optional[int] = None
    strip_tokens: str = ""
    logical_shards: int = 1024
    num_workers: int = 1

    # training spec
    batch_size: int = 2
    acc_steps: int = 1
    num_steps: int = 1000000
    training_stage: str = "initial"
    learning_rate: float = 3e-4
    grad_clip_thresh: float = 1.0
    seed: int = 2023

    # logging
    report_interval: int = 100
    tracker: Optional[str] = None  # None, "wandb", "aim"
    tracker_dir: str = "/fsx/aim_logs/llama"
    tracker_project_name: str = "llama"  # project name for a group of runs
    tracker_run_id: Optional[str] = None  # run id, for job resume purpose

    # compile
    use_torch_compile: bool = True


def causal_lm(data_seq, prompt_len=1):
    """
    Perform causal language modeling by right-shifting the input sequence.
    Sets first prompt_len tokens to be ignored by the loss.
    """
    data_seq = torch.tensor(data_seq, dtype=torch.int)
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
    if cfg.tracker:
        if cfg.tracker not in ["wandb", "aim"]:
            raise ValueError(f"tracker {cfg.tracker} not supported.")
        tracker_dir = cfg.tracker_dir
        project_name = cfg.tracker_project_name
        run_id = cfg.tracker_run_id

        if cfg.tracker == "wandb":
            try:
                import wandb  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to wandb but wandb is not installed.")
            print("--> wandb is enabled!")
            try:
                wandb.init(
                    project=project_name,
                    dir=tracker_dir,
                    resume="allow",
                    id=run_id,
                )
            except wandb.errors.UsageError:
                raise ValueError(
                    "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                )
            wandb.config = asdict(cfg)

    model.train()

    start = time.time()
    loop_start = time.time()
    loss_history = []

    # Each batch is comprised of acc_steps mini batches. acc_steps backwards per optim step.
    for mini_batch_idx, (input, label) in enumerate(train_loader, start=1):
        batch_idx, acc_step_idx = divmod(mini_batch_idx, cfg.acc_steps)
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
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx % cfg.report_interval == 0) and is_last_mini_batch:
            with torch.no_grad():
                train_loss = torch.stack(loss_history).mean()
                loss_history.clear()
            elapsed_time = time.time() - loop_start
            tokens_seen = batch_idx * cfg.batch_size * cfg.seq_length * cfg.acc_steps
            total_tokens_seen = tokens_seen
            current_loss = train_loss.item()
            current_lr = scheduler.get_last_lr()[0]
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

            print("step:", batch_idx)
            print("loss:", current_loss)
            print("LR:", current_lr)
            print("tokens seen:", total_tokens_seen)
            print("reserved memory:", reserved_mem)
            print("allocated memory:", allocated_mem)
            print("current step time:", current_step_time)
            print("overall step time:", overall_step_time)
            print("current token per gpu per sec:", current_throughput)
            print("overall token per gpu per sec:", overall_throughput)
            print(
                "overall token per day:",
                int(tokens_seen / elapsed_time * 3600 * 24),
            )
            if cfg.tracker:
                vals_to_track = {
                    "learning rate": current_lr,
                    "loss": current_loss,
                    "token seen": total_tokens_seen,
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

    return train_loss


def main(**kwargs):
    # get configs
    cfg = mup_config()
    update_config(cfg, **kwargs)
    print(f"{cfg=}")

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"--> running with these configs {cfg}")

    # some setups
    # setup()
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    setup_environ_flags()
    os.environ["TRITON_CACHE_DIR"] = os.path.join(Path.home(), ".triton", "cache")

    # get model
    # config_data = get_model_config(cfg.model_variant)
    # mamba_config = MambaConfig(**config_data)
    # model = MambaLMHeadModel(mamba_config)
    model, mamba_config = get_transformer_and_config(
        width=cfg.width,
        n_layer=cfg.n_layer,
        vocab_size=cfg.vocab_size,
        head_dim=cfg.head_dim,
        device="cuda",
        mup=cfg.mup,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6} Million params\n")
    print(f"{model=}")

    # get data loader
    print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg)
    else:
        train_loader = get_dummy_loader(cfg)
    print("Datasets constructed!")

    # torch compile
    if cfg.use_torch_compile:
        print("--> enabling torch compile...")
        # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
        torch._dynamo.config.accumulated_cache_size_limit = 128
        model = torch.compile(model)

    # Model init and Optimizer
    if cfg.mup:
        apply_mup_init(model)
        optimizer = optim.AdamW(
            get_mup_optim_iter(model, lr=cfg.learning_rate, optim_type="adam"),
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

    # Train
    print(f"Training for {cfg.num_steps} steps")
    train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
    )

    # checkpointer.save_single_file(cfg.num_steps, model)


if __name__ == "__main__":
    fire.Fire(main)
