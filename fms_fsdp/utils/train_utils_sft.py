import os
from dataclasses import asdict
from functools import partial
from warnings import warn

import torch
import torch.distributed._functional_collectives as funcol

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from fms_fsdp import config
from fms_fsdp.policies import *
from fms_fsdp.utils.checkpointing_utils_sft import save_hf_model
from fms_fsdp.utils.dataloader_utils_sft import DatasetStats


def parse_args(x):
    if isinstance(x, str):
        return [item.strip() for item in x.split(",")]
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, (int, float, complex)):
        return [x]
    raise ValueError(f"arg input {x} cannot be parsed.")


# TODO: @goon - cache this value
def num_epochs_completed(
    cfg: config.train_config,
    dataset_stats: DatasetStats,
    local_rank: int,
    dp_mesh,
) -> float:
    """
    Working def for number of completed epochs:
    * All-reduce sum the number of examples seen for each dataset.
    * Divide by number of expected examples per epoch per dataset, accounting for weights
    * Take the minimum (but non-zero value) over datasets, so that we don't cut short.
    """

    weights_t = torch.tensor(parse_args(cfg.weights), dtype=torch.float32).to(
        local_rank
    )
    # Normalize weights so that the largest weight is 1, so that these weights multiplied by the
    # dataset length give the natural definition for the number of examples per epoch per dataset.
    weights_t /= weights_t.max()
    examples_per_epoch = dataset_stats.dataset_lens.to(local_rank) * weights_t
    if dp_mesh is not None:
        examples_seen_t = funcol.all_reduce(
            dataset_stats.examples_seen.to(local_rank),
            reduceOp="sum",
            group=dp_mesh.get_group(),
        )
        examples_seen_t.wait()
    else:
        examples_seen_t = dataset_stats.examples_seen.to(local_rank)
    epochs_completed_per_dataset = examples_seen_t / examples_per_epoch
    # Get the minimum of the non-zero entries. Avoiding the non-zero cases to avoid later divisions
    # by zero. This can only happen super early in training, for typical cases.
    epochs_completed = (
        epochs_completed_per_dataset[epochs_completed_per_dataset > 0].min().item()
    )
    return epochs_completed


def should_stop_training(
    step_idx: int,
    cfg: config.train_config,
    dataset_stats: DatasetStats,
    local_rank: int,
    dp_mesh,
) -> bool:
    if cfg.num_steps is not None:
        return step_idx > cfg.num_steps
    if cfg.num_epochs is not None:
        return (
            num_epochs_completed(cfg, dataset_stats, local_rank, dp_mesh)
            > cfg.num_epochs
        )
    raise ValueError(
        f"Exactly one of {cfg.num_steps=} and {cfg.num_epochs=} must be non-None"
    )


def approx_remaining_steps(
    step_idx: int,
    cfg: config.train_config,
    dataset_stats: DatasetStats,
    local_rank: int,
    dp_mesh,
) -> int:
    if cfg.num_steps is not None:
        return cfg.num_steps - step_idx + 1
    if cfg.num_epochs is not None:
        approx_epochs_seen = num_epochs_completed(
            cfg, dataset_stats, local_rank, dp_mesh
        )
        remaining_epochs = cfg.num_epochs - approx_epochs_seen
        approx_steps_per_epoch = step_idx / approx_epochs_seen
        approx_remaining_steps = int(remaining_epochs * approx_steps_per_epoch)
        return approx_remaining_steps
    raise ValueError(
        f"Exactly one of {cfg.num_steps=} and {cfg.num_epochs=} must be non-None"
    )


def approx_total_train_steps(
    step_idx: int,
    cfg: config.train_config,
    dataset_stats: DatasetStats,
    local_rank: int,
    dp_mesh,
) -> int:
    if cfg.num_steps is not None:
        return cfg.num_steps
    if cfg.num_epochs is not None:
        return step_idx + approx_remaining_steps(
            step_idx, cfg, dataset_stats, local_rank, dp_mesh
        )
    raise ValueError(
        f"Exactly one of {cfg.num_steps=} and {cfg.num_epochs=} must be non-None"
    )


def approx_frac_training_complete(
    step_idx: int,
    cfg: config.train_config,
    dataset_stats: DatasetStats,
    local_rank: int,
    dp_mesh,
) -> float:
    if cfg.num_steps is not None:
        return step_idx / cfg.num_steps
    if cfg.num_epochs is not None:
        approx_epochs_seen = num_epochs_completed(
            cfg, dataset_stats, local_rank, dp_mesh
        )
        return approx_epochs_seen / cfg.num_epochs
    raise ValueError(
        f"Exactly one of {cfg.num_steps=} and {cfg.num_epochs=} must be non-None"
    )


def train(
    cfg,
    mamba_config,
    model,
    tokenizer,
    local_rank,
    rank,
    cp_degree,
    dp_mesh,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    checkpointer,
    start_step,
    tokens_seen,
    pred_tokens_seen,
    hf_config,
    dataset_lens: list[int],
    data_schedule,
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
    # ddp_stats:
    # 0: loss
    # 1: grad
    # 2: num fwd/bwd passes (!= n_optim_steps when grad_acc != 1)
    # 3: n_toks: total sequence length (not counting padding)
    # 4: n_pred_toks: number of actual tokens which are predicted
    # 5: batch_size: for testing get_infinite_cp_batching_iter perf
    # 6: n_toks_padded: total sequence length (counting padding)
    ddp_stats = torch.zeros(7).to(local_rank)
    dataset_lens_t = torch.tensor(dataset_lens).to(local_rank)

    start = time.time()
    loop_start = time.time()
    train_loss = -1

    train_loader.seq_length = data_schedule(start_step)
    for batch_idx, (dataset_stats, batch_size, batch) in enumerate(
        train_loader, start=start_step * cfg.grad_accum_steps + 1
    ):
        input, label = batch["input_ids"], batch["labels"]
        step_idx = (batch_idx + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
        train_loader.seq_length = data_schedule(step_idx)
        should_step = batch_idx % cfg.grad_accum_steps == 0

        stop_training = should_stop_training(
            step_idx, cfg, dataset_stats, local_rank, dp_mesh
        )
        frac_complete = approx_frac_training_complete(
            step_idx, cfg, dataset_stats, local_rank, dp_mesh
        )
        remaining_steps = approx_remaining_steps(
            step_idx, cfg, dataset_stats, local_rank, dp_mesh
        )
        approx_num_steps = approx_total_train_steps(
            step_idx, cfg, dataset_stats, local_rank, dp_mesh
        )

        if stop_training:
            if not cfg.skip_ckpt and (
                cfg.checkpoint_interval == -1
                or (step_idx - 1) % cfg.checkpoint_interval != 0
            ):
                # Save before breaking, if a we didn't save last step
                save(
                    checkpointer=checkpointer,
                    step_idx=step_idx - 1,
                    model=model,
                    optimizer=optimizer,
                    tokens_seen=tokens_seen,
                    new_tokens_seen=new_tokens_seen,
                    pred_tokens_seen=pred_tokens_seen,
                    new_pred_tokens_seen=new_pred_tokens_seen,
                    tokenizer=tokenizer,
                    hf_config=hf_config,
                    rank=rank,
                    is_compiled=cfg.use_torch_compile,
                )
            break
        input = input.to(local_rank)
        label = label.to(local_rank)

        # Print out examples to sanity check:
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
            # predictions. Might not be the right thing to do, but otherwise the z-loss impact would
            # be very sensitive to the pred_token/total_token ratio.
            pred_idxs = label != -100
            if pred_idxs.any():
                z_loss_tensor = torch.logsumexp(output[pred_idxs], dim=-1).pow(2)
                if cfg.sft_loss_type == "sum":
                    loss = loss + cfg.z_loss * z_loss_tensor.sum()
                elif cfg.sft_loss_type == "mean":
                    loss = loss + cfg.z_loss * z_loss_tensor.mean()
                del z_loss_tensor

        # Avoid logits memory leak
        del output

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
        #     (loss / cfg.grad_accum_steps).backward()
        # elif cfg.sft_loss_type == "sum":
        #     (loss * world_size).backward()

        (loss / cfg.grad_accum_steps).backward()

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
        ddp_stats[5] += batch_size  # batch_size
        ddp_stats[6] += input.numel()  # n_tok_sum_padded
        if not should_step:
            continue

        # If grad_clip_thresh < 0, set the threshold to infinity so that we don't actually clip, but
        # still collect norm stats
        grad_norm = model.clip_grad_norm_(
            cfg.grad_clip_thresh if cfg.grad_clip_thresh > 0.0 else float("inf")
        )
        ddp_stats[1] += grad_norm.item()
        if not cfg.skip_optim_step:
            if not torch.isnan(grad_norm):
                optimizer.step()
            else:
                warn(
                    f"Skipping optim {step_idx=} on {rank=} due to nan grad norm.",
                    stacklevel=1,
                )
        del grad_norm
        scheduler.step(num_steps=approx_num_steps)

        if profiler:
            profiler.step()

        if step_idx == 1 or stop_training or step_idx % cfg.report_interval == 0:
            dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            # num fwd/bwd passes summed over all ranks
            n_fwd_bwd_passes = ddp_stats[2].item()
            # Total num examples seen in reporting period. Each example is over-counted by
            # cp_degree, so correct for that.
            n_examples = ddp_stats[5].item() / cp_degree
            n_optim_steps = cfg.report_interval * world_size
            g_norm = ddp_stats[1] / n_fwd_bwd_passes

            elapsed_time = time.time() - loop_start
            n_tok_sum = ddp_stats[3].item()
            n_tok_sum_padded = ddp_stats[6].item()
            padding_fraction = (n_tok_sum_padded - n_tok_sum) / n_tok_sum_padded
            n_pred_tok_sum = ddp_stats[4].item()

            if cfg.sft_loss_type == "sum":
                # This is the closest analogue of the usual mean grad norm for sum losses
                current_gnorm_per_pred_tok = (ddp_stats[1] / n_pred_tok_sum).item()

            avg_tok_per_example = n_tok_sum / n_examples
            avg_pred_tok_per_example = n_pred_tok_sum / n_examples
            # Cases:
            # 1) sft_loss_type == "sum": we compute the sum of the losses over all ranks, averaged
            #    over the number of optimizer steps. This scales with the global batch size, and so
            #    we also compute the average of this loss over the number of non-trivial pred toks.
            # 2) sft_loss_type == "mean": straight average over all ranks and steps
            if cfg.sft_loss_type == "sum":
                train_loss = ddp_stats[0] / n_optim_steps
                train_loss_per_pred_tok = ddp_stats[0].item() / n_pred_tok_sum
                train_loss_per_total_tok = ddp_stats[0].item() / n_tok_sum
            elif cfg.sft_loss_type == "mean":
                train_loss = ddp_stats[0] / n_fwd_bwd_passes
            avg_batch_size = n_examples / cfg.report_interval
            # tok_per_gpu: number of tokens seen by each GPU on average per optim step
            tok_per_gpu = int(n_tok_sum / world_size / cfg.report_interval)
            new_tokens_seen += int(n_tok_sum)
            new_pred_tokens_seen += int(n_pred_tok_sum)

            # Reduce the DatasetStats attrs, if needed:
            if dp_mesh is not None:
                dataset_epoch_idx = funcol.all_reduce(
                    dataset_stats.epoch_idx.to(local_rank),
                    reduceOp="max",
                    group=dp_mesh.get_group(),
                )
                dataset_tokens_seen = funcol.all_reduce(
                    dataset_stats.tokens_seen.to(local_rank),
                    reduceOp="sum",
                    group=dp_mesh.get_group(),
                )
                dataset_pred_tokens_seen = funcol.all_reduce(
                    dataset_stats.pred_tokens_seen.to(local_rank),
                    reduceOp="sum",
                    group=dp_mesh.get_group(),
                )
                dataset_examples_seen = funcol.all_reduce(
                    dataset_stats.examples_seen.to(local_rank),
                    reduceOp="sum",
                    group=dp_mesh.get_group(),
                )
                dataset_epoch_idx.wait()
                dataset_tokens_seen.wait()
                dataset_pred_tokens_seen.wait()
                dataset_examples_seen.wait()
            else:
                dataset_epoch_idx = dataset_stats.epoch_idx.to(local_rank)
                dataset_tokens_seen = dataset_stats.tokens_seen.to(local_rank)
                dataset_pred_tokens_seen = dataset_stats.pred_tokens_seen.to(local_rank)
                dataset_examples_seen = dataset_stats.examples_seen.to(local_rank)

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

                remaining_secs = remaining_steps * current_step_time
                print("\nstep:", step_idx)
                print("loss:", current_loss)
                if cfg.sft_loss_type == "sum":
                    print("avg loss per pred tok:", train_loss_per_pred_tok)
                    print("avg loss per total tok:", train_loss_per_total_tok)
                print(f"average global batch size: {avg_batch_size}")
                print(f"average tokens per example: {avg_tok_per_example}")
                print(f"average pred tokens per example: {avg_pred_tok_per_example}")
                print(f"allocated memory: {allocated_mem / 2**30:.2f} GiB")
                print("current examples:", n_examples)
                print("current pred toks:", n_pred_tok_sum)
                print("current step time:", current_step_time)
                print("current token per gpu per sec:", current_throughput)
                print("current token seen:", n_tok_sum)
                print("current token seen with padding:", n_tok_sum_padded)
                print(f"current tokens/step: {world_size * tok_per_gpu}")
                print("gradient norm:", current_gnorm)
                if cfg.sft_loss_type == "sum":
                    print("gradient norm per pred tok:", current_gnorm_per_pred_tok)
                print("LR:", current_lr)
                print(
                    "overall pred token per day:",
                    int(new_pred_tokens_seen / elapsed_time * 3600 * 24),
                )
                print("overall step time:", overall_step_time)
                print(
                    "overall token per day:",
                    int(new_tokens_seen / elapsed_time * 3600 * 24),
                )
                print("overall token per gpu per sec:", overall_throughput)
                print("padding fraction:", padding_fraction)
                print("pred tokens seen:", total_pred_tokens_seen)
                print(f"remaining steps: {remaining_steps}")
                print(f"reserved memory: {reserved_mem / 2**30:.2f} GiB")
                print("tokens seen:", total_tokens_seen)
                print(f"{dataset_epoch_idx=}")
                print(f"{dataset_tokens_seen=}")
                print(f"{dataset_pred_tokens_seen=}")
                print(f"{dataset_examples_seen=}")

                fraction_dataset_seen = dataset_examples_seen / dataset_lens_t
                print(f"{fraction_dataset_seen=}")
                print(
                    f"Expected epochs per dataset: {fraction_dataset_seen / frac_complete}"
                )
                print(
                    f"Expected tok per dataset: {dataset_tokens_seen / frac_complete}"
                )
                print(
                    f"Expected pred tok per dataset: {dataset_pred_tokens_seen / frac_complete}"
                )
                print(
                    f"Expected examples per dataset: {dataset_examples_seen / frac_complete}"
                )
                if cfg.checkpoint_interval == -1:
                    steps_until_ckpt = remaining_steps
                else:
                    next_ckpt_step_idx = (
                        (step_idx + cfg.checkpoint_interval - 1)
                        // cfg.checkpoint_interval
                    ) * cfg.checkpoint_interval
                    steps_until_ckpt = next_ckpt_step_idx - step_idx
                secs_until_ckpt = steps_until_ckpt * current_step_time
                print(
                    f"Approx. time to next ckpt: {timedelta(seconds=secs_until_ckpt)}"
                )
                print(f"Approx. time remaining: {timedelta(seconds=remaining_secs)}")

                if cfg.tracker:
                    vals_to_track = {
                        "data/avg toks per example": avg_tok_per_example,
                        "data/avg pred toks per example": avg_pred_tok_per_example,
                        "data/avg global bsz": avg_batch_size,
                        "data/current num examples": n_examples,
                        "data/current pred toks": n_pred_tok_sum,
                        "data/current toks seen with padding": n_tok_sum_padded,
                        "data/current toks seen": n_tok_sum,
                        "data/padding fraction": padding_fraction,
                        "data/pred toks seen": total_pred_tokens_seen,
                        "data/toks seen": total_tokens_seen,
                        "perf/current throughput (toks per gpu per sec)": current_throughput,
                        "perf/overall throughput (toks per gpu per sec)": overall_throughput,
                        "perf/gpu allocated memory": allocated_mem,
                        "perf/gpu reserved memory": reserved_mem,
                        "gradient norm": current_gnorm,
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "frac_complete": frac_complete,
                    }
                    # Individual dataset stats
                    for dset_idx, tok_seen in enumerate(dataset_tokens_seen.tolist()):
                        vals_to_track[f"data/dataset_{dset_idx} toks seen"] = tok_seen
                    for dset_idx, pred_tok_seen in enumerate(
                        dataset_pred_tokens_seen.tolist()
                    ):
                        vals_to_track[f"data/dataset_{dset_idx} pred toks seen"] = (
                            pred_tok_seen
                        )
                    for dset_idx, ex_seen in enumerate(dataset_examples_seen.tolist()):
                        vals_to_track[f"data/dataset_{dset_idx} examples seen"] = (
                            ex_seen
                        )
                    for dset_idx, frac_seen in enumerate(
                        fraction_dataset_seen.tolist()
                    ):
                        vals_to_track[f"data/dataset_{dset_idx} fraction seen"] = (
                            frac_seen
                        )

                    if cfg.sft_loss_type == "sum":
                        vals_to_track["loss_per_pred_tok"] = train_loss_per_pred_tok
                        vals_to_track["loss_per_total_tok"] = train_loss_per_total_tok
                        vals_to_track["gradient norm per pred tok"] = (
                            current_gnorm_per_pred_tok
                        )
                    if cfg.tracker == "wandb":
                        tracker_fn = wandb.log
                    elif cfg.tracker == "aim":
                        tracker_fn = run.track
                    tracker_fn(vals_to_track, step=step_idx)

            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if (
            not cfg.skip_ckpt
            and cfg.checkpoint_interval != -1
            and step_idx % cfg.checkpoint_interval == 0
        ):
            save(
                checkpointer=checkpointer,
                step_idx=step_idx,
                model=model,
                optimizer=optimizer,
                tokens_seen=tokens_seen,
                new_tokens_seen=new_tokens_seen,
                pred_tokens_seen=pred_tokens_seen,
                new_pred_tokens_seen=new_pred_tokens_seen,
                tokenizer=tokenizer,
                hf_config=hf_config,
                rank=rank,
                is_compiled=cfg.use_torch_compile,
            )

    return train_loss


def setup(cfg):
    dist.init_process_group("nccl", timeout=timedelta(seconds=cfg.pg_timeout_s))


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


def save(
    checkpointer,
    step_idx: int,
    model,
    optimizer,
    tokens_seen: int,
    new_tokens_seen: int,
    pred_tokens_seen: int,
    new_pred_tokens_seen: int,
    tokenizer,
    hf_config,
    rank,
    is_compiled: bool = False,
) -> None:
    if not rank:
        print("Saving fms-fsdp checkpoint...")
    checkpointer.save(
        step_idx,
        model,
        optimizer,
        None,
        tokens_seen=tokens_seen + new_tokens_seen,
        pred_tokens_seen=pred_tokens_seen + new_pred_tokens_seen,
    )
    model_state_dict_fms = checkpointer.get_full_state_dict(
        model, is_compiled=is_compiled
    )

    hf_save_time = time.time()
    hf_output_dir = os.path.join(
        checkpointer.ckp_path[:-12], "hf", "step_" + str(step_idx)
    )
    if not rank:
        print("Saving HF checkpoint...")
    if rank != 0:
        dist.barrier()
    else:
        save_hf_model(
            hf_config=hf_config,
            fms_state_dict=model_state_dict_fms,
            output_dir=hf_output_dir,
            tokenizer=tokenizer,
            precision="fp32",
        )
        checkpointer.report(
            f"HF checkpoint saved in {hf_output_dir}",
            hf_save_time=time.time() - hf_save_time,
        )
        dist.barrier()
