import os
from dataclasses import asdict
from functools import partial

import torch

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
from fms_fsdp.utils.checkpointing_utils_sft import save_as_single_hf_safetensors_file


def train(
    cfg,
    mamba_config,
    model,
    tokenizer,
    local_rank,
    rank,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
    pred_tokens_seen,
):
    if cfg.sft_loss_type not in ("sum", "mean"):
        raise ValueError(f"{cfg.sft_loss_type=} not mean or sum")

    world_size = int(os.environ["WORLD_SIZE"])
    new_tokens_seen = 0
    new_pred_tokens_seen = 0
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
                        config=asdict(cfg)
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

    for batch_idx, (epoch_idx, batch) in enumerate(
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
            input_toks_list = input.cpu().tolist()
            for example_idx, toks in enumerate(input_toks_list):
                print(
                    f"[{rank=}, {batch_idx=}, {example_idx=} INPUTS]:  {tokenizer.decode(toks)}"
                )
            label_toks_list = label.cpu().tolist()
            for example_idx, toks in enumerate(label_toks_list):
                # The -100 masking label is not a valid token, so just take the abs to make it one
                toks = [abs(t) for t in toks]
                print(
                    f"[{rank=}, {batch_idx=}, {example_idx=} LABELS]:  {tokenizer.decode(toks)}"
                )

        optimizer.zero_grad()
        output = model(input)
        output = output.logits if hasattr(output, "logits") else output
        if cfg.sanity_print_toks:
            for batch_idx in range(label.shape[0]):
                idxs = label[batch_idx] != -100
                if not torch.any(idxs):
                    print(f"[{rank=}, {batch_idx=}]: no preds!")
                else:
                    gold_labels = label[batch_idx][idxs]
                    preds = output[batch_idx][idxs].max(dim=-1).indices
                    print(
                        f"[{rank=}, {batch_idx=}]:\n\tLabel: {tokenizer.decode(gold_labels)}\n\tPreds: {tokenizer.decode(preds)}"
                    )
                    del gold_labels
                    del preds

        loss = ce_loss(output.view(-1, output.size(-1)), label.reshape(-1).long())

        if cfg.z_loss is not None:
            # NOTE: @goon - only applying z-loss to the tokens corresponding to non-trivial
            # predictions. Might not be the right thing to do.
            pred_idxs = label != -100
            if pred_idxs.any():
                z_loss_tensor = torch.logsumexp(output[pred_idxs], dim=-1).pow(2)
                if cfg.sft_loss_type == "sum":
                    loss = loss + cfg.z_loss * z_loss_tensor.sum()
                elif cfg.sft_loss_type == "mean":
                    loss = loss + cfg.z_loss * z_loss_tensor.mean()
                del z_loss_tensor

        # # NOTE: @goon - the below is what is strictly needed for correctness, but it's not what
        # # open-instruct does. So, instead of doing the right thing, we just follow open OI to
        # # minimize differences.
        # # [Grad accumulation & FSDP averaging]
        # # 1) Mean loss: logically we are averaging over grad acc steps, so divide by the grad acc
        # #    factor before backwards.
        # # 2) Sum loss: logically we are summing over all ranks, so we both *avoid* dividing by grad
        # #    acc steps and multiply by the world size to counteract the FSDP averaging.
        # # (Note: These scaling factors largely drop out of Adam anyway. Globally re-scaling the loss
        # # via loss -> loss * X has the same effect as scaling eps -> eps / X.)
        # if cfg.sft_loss_type == "mean":
        #     (loss / cfg.grad_acc_steps).backward()
        # elif cfg.sft_loss_type == "sum":
        #     (loss * world_size).backward()

        (loss / cfg.grad_acc_steps).backward()

        # NOTE: @goon - when using a "mean" loss, the loss will be nan when if all labels are -100,
        # as is usually the case for early ranks. Count these as zeros for now. This messes up the
        # reporting a bit, but not sure what else to do?
        if torch.isnan(loss):
            ddp_stats[0] += 0.0
        else:
            ddp_stats[0] += loss.detach().item()
        ddp_stats[2] += 1  # n_fwd_bwd_passes
        ddp_stats[3] += (input != 0).sum().item()  # n_tok_sum (don't include padding!)
        ddp_stats[4] += (label != -100).sum().item()  # n_pred_toks
        if not should_step:
            continue

        # Skip clipping if grad_clip_thresh < 0
        if cfg.grad_clip_thresh > 0:
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
            avg_n_pred_toks = n_pred_tok_sum / n_fwd_bwd_passes

            # Cases:
            # 1) sft_loss_type == "sum": we compute the sum of the losses over all ranks, averaged
            #    over the number of optimizer steps. This scales with the global batch size, and so
            #    we also compute the average of this loss over the number of non-trivial pred toks.
            # 2) sft_loss_type == "mean": straight average over all ranks and steps
            if cfg.sft_loss_type == "sum":
                n_optim_steps = cfg.report_interval * world_size
                train_loss = ddp_stats[0] / n_optim_steps
                train_loss_per_pred_tok = ddp_stats[0].item() / n_pred_tok_sum
            elif cfg.sft_loss_type == "mean":
                train_loss = ddp_stats[0] / n_fwd_bwd_passes
            # tok_per_gpu: number of tokens seen by each GPU on average per optim step
            tok_per_gpu = int(n_tok_sum / world_size / cfg.report_interval)
            new_tokens_seen += int(n_tok_sum)
            new_pred_tokens_seen += int(n_pred_tok_sum)
            if rank == 0:
                total_tokens_seen = int(tokens_seen + new_tokens_seen)
                total_pred_tokens_seen = int(pred_tokens_seen + new_pred_tokens_seen)
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
                print(f"{epoch_idx=}")
                print("tokens seen:", total_tokens_seen)
                print("pred tokens seen:", total_pred_tokens_seen)
                print("current token seen:", n_tok_sum)
                print("current pred toks:", n_pred_tok_sum)
                print("avg toks preds per gpu per example:", avg_n_pred_toks)
                print(f"current tokens/step: {world_size * tok_per_gpu}")
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
                print(
                    "overall pred token per day:",
                    int(new_pred_tokens_seen / elapsed_time * 3600 * 24),
                )
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
                        "epoch": epoch_idx,
                        "pred token seen": total_pred_tokens_seen,
                        "current token seen": n_tok_sum,
                        "current pred toks": n_pred_tok_sum,
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
                pred_tokens_seen=pred_tokens_seen + new_pred_tokens_seen,
            )
            _, model_state_dict_fms = checkpointer.save_single_file(step_idx, model)

            hf_output_dir = os.path.join(
                checkpointer.ckp_path[:-12], "hf", "step_" + str(step_idx)
            )
            save_as_single_hf_safetensors_file(
                mamba_cfg=mamba_config,
                mamba_state_dict=model_state_dict_fms,
                output_dir=hf_output_dir,
                tokenizer=tokenizer,
                precision="fp32",
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
