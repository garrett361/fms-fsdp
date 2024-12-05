import torch.distributed as dist


import os


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    try:
        dist.init_process_group("gloo")
        if not rank:
            sweep_id_list = [1]
        else:
            sweep_id_list = [None]
        dist.broadcast_object_list(sweep_id_list, src=0)
        print(f"Found {sweep_id_list=} on {rank=}")
    finally:
        dist.destroy_process_group()
