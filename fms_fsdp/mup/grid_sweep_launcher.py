from typing import Any
import multiprocessing as mp
import os
from copy import deepcopy

import fire
import wandb

from single_gpu_training_transformer_only import mup_config, main, get_cfg_from_kwargs


import warnings


"""
A wandb sweep launcher. Pass a dict[str, tuple|list] --sweep_params arg which will be grid-scanned
over. Example:

```bash
LRS=$(python -c 'print([10**(-n/3) for n in range(6, 8)])')
SEEDS=$(python -c 'print(list(range(42, 44)))')
SWEEP_PARAMS="{learning_rate:$LRS,seed:$SEEDS}"
python3  grid_sweep_launcher.py --n_layer=10 --sweep_params="$SWEEP_PARAMS"
```
"""

FIRE_CLI_ARGS: dict[str, Any] = {}
SWEEP_CFG: dict[str, Any] = {
    "method": "grid",
    "metric": {"goal": "minimize", "name": "loss"},
}


def process_cli_args(**kwargs) -> None:
    assert kwargs["tracker"] == "wandb"
    global FIRE_CLI_ARGS
    global SWEEP_CFG

    # Get the bare cli args
    FIRE_CLI_ARGS = deepcopy(kwargs)

    # And then build up the wandb sweep args

    # Expect a --sweep_params arg, which provides a dict
    sweep_params = kwargs.pop("sweep_params")
    if not isinstance(sweep_params, dict):
        raise ValueError(f"Expected a dict, got {sweep_params=}")

    # All other kwargs are fixed values. wandb expects this structure:
    all_params = {k: {"value": v} for k, v in kwargs.items()}

    # Merge in the sweep config w/ correct wandb syntax
    for k, v in sweep_params.items():
        if not isinstance(k, str) or not isinstance(v, (tuple, list)):
            raise ValueError(
                f"--sweep_params should be dict[str, tuple|list] dict, found {k=}, {v=}"
            )
        if k in all_params:
            warnings.warn(f"Overwriting key {k=}, v={all_params[k]} with {v=}")
        all_params[k] = {"values": v}

    # Merge the sweep config into the fixed config.

    SWEEP_CFG["parameters"] = all_params


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
    fire.Fire(process_cli_args)
    print(f"Running sweep with config:\n{SWEEP_CFG}")

    sweep_id_queue = mp.Queue()

    def main_wrapper():
        with wandb.init(resume="never") as run:
            cfg_dict = wandb.config
            cfg = get_cfg_from_kwargs(**cfg_dict)
            print(f"Started with {cfg_dict=}")
            # Important: for some reason there are frequent hangs if we use a non-trivial id in
            # wandb.init when this script is run under mutiprocessing, but it works fine if we
            # just set the name by hand.
            run_name = create_wandb_run_id(cfg)
            run.name = run_name
            main(cfg)

    def target(device_idx: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
        # Only device 0 starts the sweep
        if device_idx == "0":
            sweep_id = wandb.sweep(
                SWEEP_CFG,
                project=FIRE_CLI_ARGS["tracker_project_name"],
            )
            sweep_id_queue.put(sweep_id)
        else:
            os.environ["WANDB_PROJECT"] = FIRE_CLI_ARGS["tracker_project_name"]
            sweep_id = sweep_id_queue.get()
            sweep_id_queue.put(sweep_id)

        wandb.agent(sweep_id, main_wrapper)

    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    processes = [
        mp.Process(target=target, args=(device_idx,)) for device_idx in devices
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
