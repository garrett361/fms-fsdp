import torch.distributed as dist


import os


from typing import Any
from copy import deepcopy

import fire
import wandb
from fms_fsdp.mup import SWEEP_FILE, PROJECT_FILE


import warnings


"""
Creates a wandb sweep from the CLI args and writes the sweep id and wandb project to node-local
files. To be follwed up with run_sweep_worker.py which starts the corresponding sweep workers.

Pass a dict[str, tuple|list] --sweep_params arg which will be grid-scanned over, in addition to the
usual other fire cli args.

Example:

```bash
#On every node: 

LRS=$(python -c 'print([10**(-n/3) for n in range(6, 8)])')
SEEDS=$(python -c 'print(list(range(42, 44)))')
SWEEP_PARAMS="{learning_rate:$LRS,seed:$SEEDS}"

# Run once on each node. node-rank-0 starts the sweep, communicates the sweep_id to all other nodes
# and the node 
torchrun --nnodes=${N_NODES} --node_rank=${NODE_RANK} --nproc_per_node=1 \ 
    <other-torchrun-flags> start_multi_node_sweep.py --n_layer=10 <other_cli_args> --sweep_params="$SWEEP_PARAMS"

# Then start workers on each node:
python3 fms_fsdp/mup/run_sweep_worker.py
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


if __name__ == "__main__":
    fire.Fire(process_cli_args)
    print(f"Running sweep with config:\n{SWEEP_CFG}")

    project = FIRE_CLI_ARGS["tracker_project_name"]
    rank = int(os.environ["RANK"])
    try:
        dist.init_process_group("gloo")
        if not rank:
            sweep_id = wandb.sweep(SWEEP_CFG, project=project)
            sweep_id_list = [sweep_id]
        else:
            sweep_id_list = [None]
        dist.broadcast_object_list(sweep_id_list, src=0)
        print(f"Found {sweep_id_list=} on {rank=}")
        with open(SWEEP_FILE, "w") as f:
            f.write(sweep_id_list[0])
        with open(PROJECT_FILE, "w") as f:
            f.write(project)
        print(f"Done write on {rank=}")

    finally:
        dist.destroy_process_group()
