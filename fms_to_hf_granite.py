import fire
import torch
from fms.models import get_model
from torch.distributed._shard.checkpoint import FileSystemReader, load_state_dict
from transformers import AutoConfig, AutoTokenizer

from fms_fsdp.utils.checkpointing_utils import save_granite_hf_model


def main(load_path, save_path, hf_cfg_path):
    hf_config = AutoConfig.from_pretrained(hf_cfg_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_cfg_path)
    model = get_model(
        "hf_pretrained",
        model_path=hf_cfg_path,
        cp_mesh=None,
        distributed_strategy="do not distribute",  # Hack
        data_type=torch.float32,
    )
    fms_state_dict = {"model_state": model.state_dict()}
    load_state_dict(
        state_dict=fms_state_dict,
        storage_reader=FileSystemReader(load_path),
        no_dist=True,
    )
    print("Saving...")
    save_granite_hf_model(
        hf_config,
        fms_state_dict,
        save_path,
        tokenizer,
    )


if __name__ == "__main__":
    fire.Fire(main)
