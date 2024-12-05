from fms_fsdp.mup import SWEEP_FILE, PROJECT_FILE
import multiprocessing as mp
import os

import wandb

from fms_fsdp.mup.single_gpu_training_transformer_only import (
    mup_config,
    main,
    get_cfg_from_kwargs,
)


"""
Starts a one single-GPU worker per available GPU to carry out a wandb sweep. Assumes the
`start_multi_node_sweep.py` file was run first, populating the SWEEP_FILE and PROJECT_FILE.
"""


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
    with open(SWEEP_FILE, "r") as f:
        sweep_id = f.read()
    with open(PROJECT_FILE, "r") as f:
        project = f.read()

    def main_wrapper():
        with wandb.init(resume="never") as run:
            cfg_dict = wandb.config
            cfg = get_cfg_from_kwargs(**cfg_dict)
            # Important: for some reason there are frequent hangs if we use a non-trivial id in
            # wandb.init when this script is run under mutiprocessing, but it works fine if we
            # just set the name by hand.
            run_name = create_wandb_run_id(cfg)
            run.name = run_name
            main(cfg)

    def test_wrapper():
        with wandb.init(resume="never") as run:
            cfg_dict = wandb.config
            cfg = get_cfg_from_kwargs(**cfg_dict)
            # Important: for some reason there are frequent hangs if we use a non-trivial id in
            # wandb.init when this script is run under mutiprocessing, but it works fine if we
            # just set the name by hand.
            run_name = create_wandb_run_id(cfg)
            run.name = run_name
            wandb.log({"step": 1}, step=1)

    def target(device_idx: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
        # HACKS: must manually set the expected WANDB_PROJECT env var.
        os.environ["WANDB_PROJECT"] = project
        wandb.agent(sweep_id, test_wrapper)

    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    processes = [
        mp.Process(target=target, args=(device_idx,)) for device_idx in devices
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
