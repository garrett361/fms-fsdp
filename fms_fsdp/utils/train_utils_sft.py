import os
from dataclasses import asdict
from functools import partial

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from fms_fsdp.policies import *


def train(
    cfg,
    model,
    local_rank,
    rank,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
    cp_degree: int = 1,
):
    world_size = int(os.environ["WORLD_SIZE"])
    new_tokens_seen = 0
    ce_loss = torch.nn.CrossEntropyLoss(reduction=cfg.sft_loss_type)
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
                    )
                except wandb.errors.UsageError:
                    raise ValueError(
                        "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                    )
                wandb.config = asdict(cfg)

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
    # ddp_stats:
    # 0: loss
    # 1: grad
    # 2: num fwd/bwd passes (!= n_optim_steps when grad_acc != 1)
    # 3: n_toks: total sequence length
    # 4: n_pred_toks: number of actual tokens which are predicted
    ddp_stats = torch.zeros(5).to(local_rank)

    start = time.time()
    loop_start = time.time()
    train_loss = -1
    if cfg.sanity_print_toks:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)

    for batch_idx, batch in enumerate(
        train_loader, start=start_step * cfg.grad_acc_steps + 1
    ):
        input, label = batch["input_ids"], batch["labels"]

        step_idx = (batch_idx + cfg.grad_acc_steps - 1) // cfg.grad_acc_steps
        should_step = batch_idx % cfg.grad_acc_steps == 0
        if step_idx > cfg.num_steps:
            break
        input = input.to(local_rank)
        label = label.to(local_rank)
        if cfg.sanity_print_toks:
            toks_list = input.cpu().tolist()
            for toks in toks_list:
                print(f"[{rank=}]:  {tokenizer.decode(toks)}")

        optimizer.zero_grad()
        output = model(input)
        output = output.logits if hasattr(output, "logits") else output

        # NOTE: @goon - need to shift the labels manually post-forward. Must be done post-forward
        # because the inputs are padded to be specifically divisible by the cp_degree and similar
        # factors.
        output_truncated = output[:, :-1]
        label_shifted = label[:, 1:]
        loss = ce_loss(
            output_truncated.view(-1, output_truncated.size(-1)),
            label_shifted.view(-1).long(),
        )

        if cfg.z_loss is not None:
            # NOTE: @goon - only applying z-loss to the tokens correspoding to non-trivial
            # predictions
            pred_idxs = label_shifted != -100
            if pred_idxs.any():
                z_loss_tensor = torch.logsumexp(output_truncated[pred_idxs], dim=-1).pow(2)
                if cfg.sft_loss_type == "sum":
                    loss = loss + cfg.z_loss * z_loss_tensor.sum()
                elif cfg.sft_loss_type == "mean":
                    loss = loss + cfg.z_loss * z_loss_tensor.mean()
                else:
                    raise ValueError(f"{cfg.sft_loss_type=} not mean or sum")

        # NOTE: @goon - FSDP1 will average the grads, whereas we would really want to sum them for a
        # sum loss. This doesn't hugely matter for AdamW, and we won't worry about it for now.
        # This also makes run with the same global_bs = world_size * batch_size * grad_acc  not
        # precisely equal, but it should be a relatively minor effect.
        loss.backward()

        # NOTE: @goon - when using a "mean" loss, the loss will be nan when if all labels are -100,
        # as is usually the case for early ranks. Count these as zeros for now. This messes up the
        # reporting a bit, but not sure what else to do?
        if torch.isnan(loss):
            ddp_stats[0] += 0.0
        else:
            ddp_stats[0] += loss.detach().item()
        ddp_stats[2] += 1  # n_fwd_bwd_passes
        ddp_stats[3] += input.numel()  # n_tok_sum
        ddp_stats[4] += (label != -100).sum().item()  # n_pred_toks
        if not should_step:
            continue

        ddp_stats[1] += model.clip_grad_norm_(cfg.grad_clip_thresh).item()
        optimizer.step()
        scheduler.step()

        if profiler:
            profiler.step()

        if step_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            # num fwd/bwd passes summed over all ranks
            n_fwd_bwd_passes = ddp_stats[2].item()
            g_norm = ddp_stats[1] / n_fwd_bwd_passes
            elapsed_time = time.time() - loop_start
            n_tok_sum = ddp_stats[3].item()
            n_pred_tok_sum = ddp_stats[4].item()

            # Cases:
            # 1) sft_loss_type == "sum": we compute the sum of the losses over all ranks,
            # averaged over the number of fwd/bwd steps *per rank*. This scales with the global
            # batch size, and so we also compute the average of this loss over the number of
            # non-trivial pred toks.
            # 2) sft_loss_type == "mean": straight average over all ranks and steps
            if cfg.sft_loss_type == "sum":
                n_fwd_bwd_passed_per_rank = n_fwd_bwd_passes / world_size
                train_loss = ddp_stats[0] / n_fwd_bwd_passed_per_rank
                train_loss_per_pred_tok = ddp_stats[0].item() / n_pred_tok_sum
            elif cfg.sft_loss_type == "mean":
                train_loss = ddp_stats[0] / n_fwd_bwd_passes
            else:
                raise ValueError(f"{cfg.sft_loss_type=} not mean or sum")

            tok_per_gpu = int(n_tok_sum / world_size / cfg.report_interval)
            new_tokens_seen += int(n_tok_sum)
            if rank == 0:
                total_tokens_seen = int(tokens_seen + new_tokens_seen)
                current_loss = train_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                current_gnorm = g_norm.item()
                current_step_time = (time.time() - start) / cfg.report_interval
                overall_step_time = elapsed_time / (step_idx - start_step)
                current_throughput = int(tok_per_gpu / current_step_time)
                overall_throughput = int(tok_per_gpu / overall_step_time)
                reserved_mem = torch.cuda.max_memory_reserved(
                    device=torch.cuda.current_device()
                )
                allocated_mem = torch.cuda.max_memory_allocated(
                    device=torch.cuda.current_device()
                )

                print("\nstep:", step_idx)
                print("loss:", current_loss)
                if cfg.sft_loss_type == "sum":
                    print("avg loss per pred tok:", train_loss_per_pred_tok)
                print("LR:", current_lr)
                print("tokens seen:", total_tokens_seen)
                print("new tokens seen:", new_tokens_seen)
                print("gradient norm:", current_gnorm)
                print(f"reserved memory: {reserved_mem / 2**30:.2f} GiB")
                print(f"allocated memory: {allocated_mem / 2**30:.2f} GiB")
                print("current step time:", current_step_time)
                print("overall step time:", overall_step_time)
                print("current token per gpu per sec:", current_throughput)
                print("overall token per gpu per sec:", overall_throughput)
                print(
                    "overall token per day:",
                    int(new_tokens_seen / elapsed_time * 3600 * 24),
                )
                print(f"Total tok/step: {world_size * tok_per_gpu}")
                remaining_steps = cfg.num_steps - step_idx + 1
                remaining_secs = remaining_steps * current_step_time
                print(f"remaining steps: {remaining_steps}")
                print(f"Approx. time remaining: {timedelta(seconds=remaining_secs)}")

                next_ckpt_step_idx = (
                    (step_idx + cfg.checkpoint_interval - 1) // cfg.checkpoint_interval
                ) * cfg.checkpoint_interval
                steps_until_ckpt = next_ckpt_step_idx - step_idx
                secs_until_ckpt = steps_until_ckpt * current_step_time
                print(
                    f"Approx. time to next ckpt: {timedelta(seconds=secs_until_ckpt)}"
                )

                if cfg.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "gradient norm": current_gnorm,
                        "token seen": total_tokens_seen,
                        "current throughput (token per gpu per sec)": current_throughput,
                        "overall throughput (token per gpu per sec)": overall_throughput,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
                    if cfg.sft_loss_type == "sum":
                        vals_to_track["loss_per_pred_tok"] = train_loss_per_pred_tok
                    if cfg.tracker == "wandb":
                        tracker_fn = wandb.log
                    elif cfg.tracker == "aim":
                        tracker_fn = run.track
                    tracker_fn(vals_to_track, step=step_idx)

            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if step_idx % cfg.checkpoint_interval == 0:
            checkpointer.save(
                step_idx,
                model,
                optimizer,
                None,
                tokens_seen=tokens_seen + new_tokens_seen,
            )

    return train_loss


def setup():
    dist.init_process_group("nccl", timeout=timedelta(seconds=60 * 60))


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


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
