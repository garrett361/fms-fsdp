import multiprocessing as mp
import os
import fire
from single_gpu_training_transformer_only import mup_config, main
from copy import deepcopy


"""
A pool based lr searcher.  Pass a --lrs="1e-3, 1e-4,..." arg along with other usual args to the cli
and each lr will be created and run.
"""

cfgs = []


def populate_cfgs(**kwargs) -> list[dict]:
    lrs = kwargs.pop("lrs")
    assert "learning_rate" not in kwargs
    base_cfg = mup_config(**kwargs)
    for lr in lrs:
        cfg = deepcopy(base_cfg)
        cfg.learning_rate = lr
        run_id = f"n_layer-{cfg.n_layer}_width-{cfg.width}_lr-{cfg.lr}"
        if cfg.mup:
            run_id = "mup_" + run_id
        cfg.tracker_run_id = run_id
        cfgs.append(cfg)


def print_cfg(x):
    print(x)


if __name__ == "__main__":
    fire.Fire(populate_cfgs)
    devices = [int(s) for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    num_devices = len(devices)
    device_idx = mp.Value("i", 0)

    def set_device() -> None:
        with device_idx.get_lock():
            curr_idx = device_idx.value
            os.environ["CUDA_VISIBLE_DEVICES"] = str(curr_idx)
            device_idx.value += 1
            device_idx.value %= num_devices

    results = []
    with mp.Pool(len(devices), initializer=set_device) as p:
        for cfg in cfgs:
            r = p.apply_async(
                main, (cfg,), error_callback=lambda error: print(f"Error: {error}")
            )
            results.append(r)

        for p_idx, r in enumerate(results):
            r.wait()
