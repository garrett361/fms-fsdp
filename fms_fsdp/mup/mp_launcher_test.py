import multiprocessing as mp
import os


def target() -> None:
    dev = os.environ["CUDA_VISIBLE_DEVICES"]
    import time

    time.sleep(3)
    print(f"Hi from {dev=}")


if __name__ == "__main__":
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
        for _ in range(2 * num_devices):
            r = p.apply_async(target)
            results.append(r)

        for p_idx, r in enumerate(results):
            r.wait()
