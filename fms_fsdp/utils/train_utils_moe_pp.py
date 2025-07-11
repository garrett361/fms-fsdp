import os
from collections import defaultdict
from dataclasses import asdict
from functools import partial

import torch
from torch import distributed as dist
from torch.distributed.tensor import DTensor

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch.cuda.nccl as nccl
from mamba_ssm.modules.block import Block
from mamba_ssm.moe_utils import (
    apply_loss_free_moe_balancing,
    attach_magnitude_hooks,
    attach_tok_count_hooks,
    clip_grad_norm_,
)
from torch.distributed.fsdp import ShardingStrategy

from fms_fsdp.policies import *


def train_moe_pp(
    cfg,
    model,
    pp_schedule,
    mesh,
    is_first,
    is_last,
    local_rank,
    rank,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
):
    world_size = int(os.environ["WORLD_SIZE"])
    if cfg.sanity_prints and not rank:
        print(os.environ)
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
            if rank == 0:
                print("--> wandb is enabled!")
                try:
                    wandb.init(
                        project=project_name,
                        dir=tracker_dir,
                        resume="allow",
                        id=run_id,
                        config=asdict(cfg),
                    )
                except wandb.errors.UsageError:
                    raise ValueError(
                        "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                    )

        if cfg.tracker == "aim":
            try:
                from aim import Run  # type: ignore
            except ImportError:
                raise ImportError("tracker is set to aim but aim is not installed.")
            if rank == 0:
                print("--> aim is enabled!")
                run = Run(
                    experiment=project_name,
                    repo=tracker_dir,
                    run_hash=run_id,
                )
                run["hparams"] = asdict(cfg)

    model.train()

    if cfg.tok_count_hooks or cfg.loss_free_balancing_lr:
        tok_count_hook_dict = attach_tok_count_hooks(model)
        tok_stats_dict = defaultdict(int)
    else:
        tok_count_hook_dict = None
        tok_stats_dict = None

    if cfg.block_mag_hooks:
        block_mag_hook_dict = attach_magnitude_hooks(model, [Block, model.lm_head])
    else:
        block_mag_hook_dict = None

    # ddp_stats
    # 0: losses
    # 1: steps
    ddp_stats = torch.zeros(2).to(local_rank)
    g_norms = []

    start = time.time()
    loop_start = time.time()
    train_loss = -1
    if cfg.extra_timing:
        raise NotImplementedError("extra_timing not yet implemented forPP")

    pp_losses_list = []
    for batch_idx, (input, label) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > cfg.num_steps:
            break
        input = input.to(local_rank)
        label = label.to(local_rank)

        optimizer.zero_grad()
        if is_first:
            pp_schedule.step(input)
        elif is_last:
            pp_schedule.step(target=label, losses=pp_losses_list)
        else:
            pp_schedule.step()

        # Not all PP shards are guaranteed to have MoE layers:
        if cfg.loss_free_balancing_lr and tok_count_hook_dict:
            tok_count_hook_dict.all_reduce(group=mesh["ep"].get_group())
            apply_loss_free_moe_balancing(
                cfg.loss_free_balancing_lr, model, tok_count_hook_dict
            )
            update_tok_stats_dict(
                tok_count_hook_dict, tok_stats_dict, cfg.ep_degree, world_size
            )
            tok_count_hook_dict.reset()

        if cfg.skip_clip:
            g_norms.append(-1.0)
        else:
            norm_t = clip_grad_norm_(
                model.parameters(), cfg.grad_clip_thresh, pp_mesh=mesh["pp"]
            )
            if isinstance(norm_t, DTensor):
                norm_t = norm_t.full_tensor()
            g_norms.append(norm_t.item())
        if not cfg.skip_optim_step:
            optimizer.step()
            scheduler.step()

        # Only adjust ddp_stats on those ranks which actually compute losses
        if pp_losses_list:
            ddp_stats[0] += torch.stack(pp_losses_list).mean().detach().item()
            ddp_stats[1] += 1
            pp_losses_list.clear()

        if profiler:
            profiler.step()

        if batch_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            train_loss = ddp_stats[0] / ddp_stats[1]
            elapsed_time = time.time() - loop_start

            # NOTE: @goon - For pp, the batch_size sets the total batch size per pipeline instance,
            # changing the throughput computation slightly.
            n_pipeline_instances = mesh.size() // mesh["pp"].size()
            total_tok_per_step = n_pipeline_instances * cfg.batch_size * cfg.seq_length
            new_tokens_seen = (batch_idx - start_step) * total_tok_per_step

            # Update tok_stats_dict if not already done
            if tok_stats_dict is not None and not tok_stats_dict:
                assert not tok_count_hook_dict.is_reduced, (
                    f"{tok_count_hook_dict=}, {tok_count_hook_dict=}"
                )
                tok_count_hook_dict.reduce(dst=0, group=mesh["ep"].get_group())
                update_tok_stats_dict(
                    tok_count_hook_dict, tok_stats_dict, cfg.ep_degree, world_size
                )

            if block_mag_hook_dict is not None:
                block_mag_hook_dict.reduce(dst=0, op=dist.ReduceOp.AVG)

            # Report per-rank cuda mem stats, since they can differ drastically by GPU.
            reserved_mem = torch.cuda.max_memory_reserved(
                device=torch.cuda.current_device()
            )
            allocated_mem = torch.cuda.max_memory_allocated(
                device=torch.cuda.current_device()
            )
            mem_tensor = torch.tensor(
                [reserved_mem, allocated_mem],
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            )
            dist.reduce(mem_tensor, dst=0, op=dist.ReduceOp.MAX)

            if rank == 0:
                total_tokens_seen = tokens_seen + new_tokens_seen
                current_loss = train_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                current_gnorm = sum(g_norms) / len(g_norms)
                current_step_time = (time.time() - start) / cfg.report_interval
                overall_step_time = elapsed_time / (batch_idx - start_step)
                # NOTE: @goon - another change in the PP throughput computation here:
                current_throughput = int(
                    total_tok_per_step / world_size / current_step_time
                )
                overall_throughput = int(
                    total_tok_per_step / world_size / overall_step_time
                )

                print("step:", batch_idx)
                print("loss:", current_loss)
                print("LR:", current_lr)
                print("tokens seen:", total_tokens_seen)
                print("gradient norm:", current_gnorm)
                print(
                    "max reserved memory (GiB):",
                    f"{mem_tensor[0].item() / 2**30:.2f}",
                )
                print(
                    "max allocated memory (GiB):",
                    f"{mem_tensor[1].item() / 2**30:.2f}",
                )
                print("current step time:", current_step_time)
                print("overall step time:", overall_step_time)
                print("current token per gpu per sec:", current_throughput)
                print("overall token per gpu per sec:", overall_throughput)
                print(
                    "overall token per day:",
                    int(new_tokens_seen / elapsed_time * 3600 * 24),
                )
                if cfg.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "gradient norm": current_gnorm,
                        "token seen": total_tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                    }
                    if tok_stats_dict is not None:
                        max_tok_count = 0
                        max_tok_fqn = None
                        min_tok_count = float("inf")
                        min_tok_fqn = None
                        for key, val in tok_stats_dict.items():
                            vals_to_track[f"hooks/tok_count/{key}"] = val
                            if "ep_rank" not in key and val > max_tok_count:
                                max_tok_count = val
                                max_tok_fqn = key
                            if "ep_rank" not in key and val < min_tok_count:
                                min_tok_count = val
                                min_tok_fqn = key

                        if cfg.sanity_prints:
                            print(f"min_tok_count ({min_tok_fqn}):", min_tok_count)
                            print(f"{tok_stats_dict=}")
                            print(f"max_tok_count ({max_tok_fqn}):", max_tok_count)
                    if block_mag_hook_dict is not None:
                        for key, val in block_mag_hook_dict.items():
                            vals_to_track[f"hooks/act_mag/{key}"] = val.value.item()

                    if cfg.tracker == "wandb":
                        tracker_fn = wandb.log
                    elif cfg.tracker == "aim":
                        tracker_fn = run.track
                    tracker_fn(vals_to_track, step=batch_idx)

            start = time.time()
            ddp_stats.zero_()

            if tok_count_hook_dict:
                tok_count_hook_dict.reset()
                tok_stats_dict.clear()
            if block_mag_hook_dict:
                block_mag_hook_dict.reset()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if not cfg.skip_ckpt and batch_idx % cfg.checkpoint_interval == 0:
            checkpointer.save(
                batch_idx,
                model,
                optimizer,
                None,
                tokens_seen=tokens_seen + new_tokens_seen,
            )

    return train_loss


def setup(cfg=None):
    pg_timeout = 60 * 60 if cfg is None or cfg.pg_timeout is None else cfg.pg_timeout
    dist.init_process_group("nccl", timeout=timedelta(seconds=pg_timeout))


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_mixed_precision_policy(cfg, rank):
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled")
    else:
        mixed_precision_policy = None

    return mixed_precision_policy


def get_policies(cfg, rank, block):
    """Get policies for mixed precision, wrapping, sharding, ac and param init function."""

    # mixed precision
    mixed_precision_policy = get_mixed_precision_policy(cfg, rank)

    # wrapping policy
    wrapping_policy = get_wrapper(block)

    # sharding strategy
    if cfg.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif cfg.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif cfg.sharding_strategy == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {cfg.sharding_strategy}")

    # ac handler
    apply_selective_ac = partial(apply_fsdp_checkpointing, block=block)

    # param init function
    if cfg.low_cpu_fsdp:
        param_init_fn = param_init_function
    else:
        param_init_fn = None

    return (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy,
        apply_selective_ac,
        param_init_fn,
    )


def get_profiler(cfg, rank):
    if not cfg.use_profiler:
        return
    if cfg.profiler_rank0_only and rank != 0:
        return
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profile_traces"),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )


class CUDATimer:
    def __init__(self, enabled: bool = True) -> None:
        self._start_events: list[torch.cuda.Event] = []
        self._stop_events: list[torch.cuda.Event] = []
        self.enabled = enabled

    def __enter__(self) -> "CUDATimer":
        if not self.enabled:
            return self
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        self._start_events.append(start)
        self._stop_events.append(stop)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if not self.enabled:
            return
        self._stop_events[-1].record()

    def __len__(self) -> int:
        return len(self._start_events)

    def get_time_list_s(self) -> list[float]:
        if not self._start_events:
            return [0.0]
        torch.cuda.synchronize()
        time_list_s = [
            start.elapsed_time(stop) / 1e3
            for start, stop in zip(self._start_events, self._stop_events)
        ]
        return time_list_s

    def get_total_time_s(self) -> float:
        return sum(self.get_time_list_s())

    def get_mean_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return sum(time_list_s) / len(time_list_s)

    def get_std_time_s(self) -> float:
        time_list_s = self.get_time_list_s()
        return torch.tensor(time_list_s).std().item()

    def reset(self) -> None:
        self._start_events.clear()
        self._stop_events.clear()


def update_tok_stats_dict(
    tok_count_hook_dict, tok_stats_dict, ep_degree: int, world_size: int
) -> None:
    for fqn, counts in tok_count_hook_dict.items():
        n_routed_experts = counts.value.numel()
        exp_per_rank = n_routed_experts // ep_degree
        dp_factor = world_size // ep_degree
        for exp_idx, tok_count in enumerate(counts.value.tolist()):
            tok_stats_dict[f"{fqn}.exp.{exp_idx}"] += tok_count
            # Really computing the avg per gpu when there's a non-trivial dp_factor.
            # TODO: @goon -  per-gpu?
            ep_rank = exp_idx // exp_per_rank
            tok_stats_dict[f"ep_rank.{ep_rank}"] += tok_count // dp_factor
