import multiprocessing as mp
import traceback
from typing import Sequence
import os
import fire
from single_gpu_training_transformer_only import mup_config, main
import wandb
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses


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
    print(f"Launching with {lrs=}")
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
            run_id += "_mup"
        cfg.tracker_run_id = run_id

        cfgs.append(cfg)


if __name__ == "__main__":
    fire.Fire(populate_cfgs)
    devices = [int(s) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    print(f"Launching on {devices=}")
    num_devices = len(devices)
    device_idx_queue = mp.Queue()
    for device_idx in devices:
        device_idx_queue.put(device_idx)

    def main_wrapper(cfg, device_idx_queue: mp.Queue):
        try:
            device_idx = device_idx_queue.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
            print(f"Using {device_idx=}")
            cfg_dict = dataclasses.asdict(cfg)
            res = main(**cfg_dict)
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
            else:
                print(f"Succeeded with {res=}")
