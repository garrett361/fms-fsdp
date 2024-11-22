import fire
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer


"""
Super minimal chat code for sanity checking
"""


def main(
    load_path,
    prefix: str = "Once upon a time, in a land far away,",
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: int = 5,
):
    print(f"Loading model from {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = MambaLMHeadModel.from_pretrained(
        load_path, device="cuda", dtype=torch.bfloat16
    )
    print(f"Loaded {model=}")

    # Define the input prefix

    # Tokenize the input prefix
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device="cuda")

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )

    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)

    # Print the generated text
    print(80 * "*")
    print(f"{prefix=}\n\n")
    print(f"{generated_text=}")


if __name__ == "__main__":
    fire.Fire(main)
