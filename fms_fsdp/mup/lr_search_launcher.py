import multiprocessing as mp
import traceback
from typing import Sequence
import os
import fire
from single_gpu_training_transformer_only import mup_config, main
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import dataclasses


"""
A pool based lr searcher.  Pass a --lrs="1e-3, 1e-4,..." arg along with other usual args to the cli
and each lr will be created and run.
"""

cfgs = []


def populate_cfgs(**kwargs) -> None:
    # --lrs  expected to be a comma separated list of numbers
    lrs = kwargs.pop("lrs")
    if not isinstance(lrs, Sequence):
        lrs = (lrs,)

    # Handle strings
    lrs = [float(eval(lr)) for lr in lrs]

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


def main_wrapper(cfg):
    try:
        cfg_dict = dataclasses.asdict(cfg)
        res = main(**cfg_dict)
        return (res, None, None)
    except Exception as e:
        return (None, e, traceback.format_exc())


if __name__ == "__main__":
    fire.Fire(populate_cfgs)
    devices = [int(s) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    print(f"Launching on {devices=}")
    num_devices = len(devices)
    device_idx = mp.Value("i", 0)

    def set_device() -> None:
        with device_idx.get_lock():
            curr_idx = device_idx.value
            os.environ["CUDA_VISIBLE_DEVICES"] = str(curr_idx)
            device_idx.value += 1
            device_idx.value %= num_devices

    # Important to use ProcessPoolExecutor, and not Pool, because multiprocessing is used in the
    # main function and Pool does not support nested mp.
    with ProcessPoolExecutor(len(devices), initializer=set_device) as p:
        for cfg, (res, err, traceback) in zip(cfgs, p.map(main_wrapper, cfgs)):
            if err:
                print(f"{cfg.learning_rate} errored with {err=}\n{traceback=}")
            else:
                print(f"{cfg.learning_rate} succeeded with {res=}")
        # for cfg, res in zip(cfgs, p.map(print_cfg, cfgs)):
        #     pass
        # try:
        #     print(f"{res=}")
        # except Exception as e:
        #     print(f"Found exception {e}")

        # cfg_dict = dataclasses.asdict(cfg)
        # r = p.map(
        #     main,
        #     kwds=cfg_dict,
        #     error_callback=lambda error: print(f"Error: {error}"),
        # )
        # results.append(r)

        # for p_idx, r in enumerate(results):
        #     r.wait()
        # for r in results:
        #     try:
        #         print(r.result())
        #     except Exception as e:
        #         print(f"Found exception {e}")
