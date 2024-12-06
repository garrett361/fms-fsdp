import multiprocessing as mp
import os

import wandb

from fms_fsdp.mup.single_gpu_training_transformer_only import (
    mup_config,
    main,
    get_cfg_from_kwargs,
)


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
    SWEEP_ID = os.environ["SWEEP_ID"]
    assert os.getenv("WANDB_PROJECT"), "Must set WANDB_PROJECT env var"

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

    def target(device_idx: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_idx
        # HACKS: must manually set the expected WANDB_PROJECT env var.
        wandb.agent(SWEEP_ID, main_wrapper)

    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    processes = [
        mp.Process(target=target, args=(device_idx,)) for device_idx in devices
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
