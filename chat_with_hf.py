import fire
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer


def main(load_path):
    print(f"Loading model from {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = MambaLMHeadModel.from_pretrained(
        load_path, device="cuda", dtype=torch.bfloat16
    )
    print(f"Loaded {model=}")

    # Define the input prefix
    prefix = "Once upon a time, in a land far away,"

    # Tokenize the input prefix
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(
        device="cuda", dtype=torch.bfloat16
    )

    # Generate text
    output = model.generate(
        input_ids,
        max_length=100,
        temperature=1,
    )

    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)

    # Print the generated text
    print(generated_text)


if __name__ == "__main__":
    fire.Fire(main)
