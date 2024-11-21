import fire
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer


def main(load_path):
    print(f"Loading model from {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = MambaLMHeadModel.from_pretrained(load_path, device="cuda")
    print(f"Loaded {model=}")


if __name__ == "__main__":
    fire.Fire(main)
