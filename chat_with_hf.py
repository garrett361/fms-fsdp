import fire
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer


def main(load_path):
    print(f"Loading model from {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = MambaLMHeadModel.from_pretrained(load_path, device="cuda")
    print(f"Loaded {model=}")

    # Define the input prefix
    prefix = "Once upon a time, in a land far away,"

    # Tokenize the input prefix
    input_ids = tokenizer.encode(prefix, return_tensors="pt").cuda()

    # Generate text
    output = model.generate(
        input_ids,
        max_length=100,  # Adjust as needed
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )

    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)

    # Print the generated text
    print(generated_text)


if __name__ == "__main__":
    fire.Fire(main)
