import multiprocessing as mp
import os
from contextlib import nullcontext
from dataclasses import asdict
from typing import Union

import fire
import wandb

import traceback
from typing import Sequence
from single_gpu_training_transformer_only import mup_config, main
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed


"""
A pool based lr searcher.  Pass a --lrs="1e-3, 1e-4,..." arg along with other usual args to the cli
and each lr will be created and run.
"""

cfgs = []


def populate_cfgs(**kwargs) -> None:
    # TODO: @goon - generalize to just passing a sweep dict

    # --lrs expected to be a comma separated list of numbers using "1e-3" type exponential notation.
    # "2**-5" exponential notation is not handled.
    lrs = kwargs.pop("lrs")
    if not isinstance(lrs, Sequence):
        lrs = (lrs,)

    assert "learning_rate" not in kwargs
    base_cfg = mup_config(**kwargs)
    for lr in lrs:
        cfg = deepcopy(base_cfg)
        cfg.learning_rate = lr
        # Create a unique id
        user_run_id = cfg.tracker_run_id or ""
        run_id = (
            user_run_id
            + ("_" if user_run_id else "")
            + f"n_layer-{cfg.n_layer}_width-{cfg.width}_lr-{cfg.learning_rate}"
            + f"_bsz-{cfg.batch_size}_acc-{cfg.acc_steps}_seq_len-{cfg.seq_length}_steps"
            + f"-{cfg.num_steps}"
        )
        if cfg.mup:
            if cfg.mup_use_width:
                run_id += f"_mup[base-{cfg.mup_base_width}-use_width]"
            else:
                run_id += f"_mup[base-{cfg.mup_base_width}]"
        cfg.tracker_run_id = run_id

        cfgs.append(cfg)


if __name__ == "__main__":
    fire.Fire(populate_cfgs)
    devices = [int(s) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    num_devices = len(devices)
    device_idx_queue = mp.Queue()
    for device_idx in devices:
        device_idx_queue.put(device_idx)

    def main_wrapper(cfg):
        try:
            device_idx = device_idx_queue.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)

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
                res = main(cfg)
            return (res, None, None)
        except Exception as e:
            return (None, e, traceback.format_exc())
        finally:
            device_idx_queue.put(device_idx)

    # https://docs.wandb.ai/guides/track/log/distributed-training/#spawn-process
    wandb.setup()

    # Important to use ProcessPoolExecutor, and not Pool, because multiprocessing is used in the
    # main function and Pool does not support nested mp.
    with ProcessPoolExecutor(len(devices)) as executor:
        futures = [executor.submit(main_wrapper, cfg) for cfg in cfgs]
        for f in as_completed(futures):
            res, err, tb = f.result()
            if err:
                print(f"Errored with {err=}\n{tb}")
