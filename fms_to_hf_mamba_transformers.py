import fire
from transformers.models.bamba.convert_mamba_ssm_checkpoint import (
    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file,
)


def main(load_path, save_path, tokenizer_name_or_path):
    # NOTE: @goon - from Naigang
    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
        load_path, "fp32", save_path, tokenizer_name_or_path
    )


if __name__ == "__main__":
    fire.Fire(main)

