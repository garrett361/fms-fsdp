import multiprocessing as mp
import os
from typing import Any

import fire
import wandb

import traceback
from single_gpu_training_transformer_only import get_cfg_from_kwargs, mup_config, main
from concurrent.futures import ProcessPoolExecutor, as_completed


"""
A wandb sweep launcher
A pool based lr searcher.  Pass a --lrs="1e-3, 1e-4,..." arg along with other usual args to the cli
and each lr will be created and run.
"""

SWEEP_CFG: dict[str, Any] = None


def populate_sweep_cfg(**kwargs) -> None:
    global SWEEP_CFG
    assert kwargs["tracker"] == "wandb"
    SWEEP_CFG = kwargs


def create_wandb_run_id(cfg: mup_config) -> str:
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
    return run_id


if __name__ == "__main__":
    fire.Fire(populate_sweep_cfg)

    sweep_id = wandb.sweep(SWEEP_CFG, project=SWEEP_CFG["tracker_project_name"])

    devices = [int(s) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    num_devices = len(devices)
    device_idx_queue = mp.Queue()
    for device_idx in devices:
        device_idx_queue.put(device_idx)

    def main_wrapper(cfg_dict):
        try:
            device_idx = device_idx_queue.get()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
            cfg = get_cfg_from_kwargs(**cfg_dict)

            ctx = wandb.init(
                project=cfg.tracker_project_name,
                dir=cfg.tracker_dir,
                resume="never",
                id=None,
                config=cfg_dict,
            )
            with ctx as run:
                # Important: for some reason there are frequent hangs if we use a non-trivial id in
                # wandb.init when this script is run under mutiprocessing, but it works fine if we
                # just set the name by hand.
                run.name = create_wandb_run_id(cfg)
                res = main(cfg)
            return (res, None, None)
        except Exception as e:
            return (None, e, traceback.format_exc())

    # https://docs.wandb.ai/guides/track/log/distributed-training/#spawn-process
    wandb.setup()

    # Important to use ProcessPoolExecutor, and not Pool, because multiprocessing is used in the
    # main function and Pool does not support nested mp.
    with ProcessPoolExecutor(len(devices)) as executor:
        futures = [
            executor.submit(wandb.agent, sweep_id, main_wrapper) for _ in devices
        ]
        for f in as_completed(futures):
            pass
            # res, err, tb = f.result()
            # if err:
            #     print(f"Errored with {err=}\n{tb}")
