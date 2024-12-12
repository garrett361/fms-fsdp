import multiprocessing as mp
import torch
import os

import wandb

from fms_fsdp.mup.multi_gpu_training_transformer_only import (
    main,
    setup,
    get_world_size,
)
from fms_fsdp.mup import get_cfg_from_kwargs

"""
Runs multi-gpu sweep workers within a node via nested mutiprocessing.
"""

if __name__ == "__main__":
    SWEEP_ID = os.environ["SWEEP_ID"]
    assert os.getenv("WANDB_PROJECT"), "Must set WANDB_PROJECT env var"
    world_size = int(os.getenv("WORLD_SIZE"))
    assert world_size is not None, "Must set WORLD_SIZE env var"
    n_avail_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    n_proc_groups, remainder = divmod(n_avail_devices, world_size)
    if remainder:
        raise ValueError(f"{world_size} must perfectly divide the {n_avail_devices=}")

    proc_group_ranks = torch.arange(n_avail_devices).reshape(n_proc_groups, -1)
    cuda_vis_devices_list = [
        ",".join(str(r) for r in ranks.tolist()) for ranks in proc_group_ranks
    ]

    def main_wrapper() -> None:
        def target(rank: int) -> None:
            # The wandb context only needs to be started on the reporting rank
            with wandb.init() as run:
                cfg_dict = wandb.config
                cfg = get_cfg_from_kwargs(**cfg_dict)
                if int(cfg.world_size) != get_world_size():
                    raise ValueError(
                        f"Mismatch: {cfg.world_size=} while {get_world_size()=}"
                    )
                setup(cfg, rank)
                print(f"Setup done on {rank=}, {os.environ['RANK']=}")
                # Important: for some reason there are frequent hangs if we use a non-trivial id in
                # wandb.init when this script is run under mutiprocessing, but it works fine if we
                # just set the name by hand.
                if cfg.tracker == "wandb" and not rank:
                    run.name = cfg.tracker_run_id
                main(cfg)

        processes = [
            mp.Process(target=target, args=(rank,)) for rank in range(get_world_size())
        ]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def target(cuda_vis_devices: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_vis_devices
        # HACKS: must manually set the expected WANDB_PROJECT env var.
        wandb.agent(SWEEP_ID, main_wrapper)

    print(f"Launching {len(cuda_vis_devices_list)} { world_size= } processes.")

    processes = [
        mp.Process(target=target, args=(vis_devices,))
        for vis_devices in cuda_vis_devices_list
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
