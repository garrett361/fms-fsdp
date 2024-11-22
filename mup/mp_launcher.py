import multiprocessing as mp
import os
from typing import Union


def target(rank: Union[str, int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    print(f"Hi from {rank=}")


if __name__ == "__main__":
    processes = [mp.Process(target=target, args=(rank,)) for rank in range(8)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
