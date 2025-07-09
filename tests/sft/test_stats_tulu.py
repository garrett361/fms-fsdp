import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES, encode_sft_example

tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")
tokenizer.chat_template = CHAT_TEMPLATES["tulu"]

print("LOAD")
train_dataset = load_dataset(
    "json",
    data_files="/datasets/instruct_data/Tuluv3/tuluv3_data.jsonl",
    num_proc=32,
)["train"]

torch.manual_seed(42)
n_samples = 5000
rand_idxs = list(random.sample(range(len(train_dataset)), n_samples))
actual_dataset_len = len(train_dataset)
train_dataset = train_dataset.select(rand_idxs)

print("MAP")


def count_toks(example):
    """
    Counts both the number of total tokens and number of non-masked predicted tokens.
    """
    toks = encode_sft_example(example, tokenizer=tokenizer, max_seq_length=2**30)
    return {
        "n_toks": toks["input_ids"].numel(),
        "n_pred_toks": (toks["labels"] != -100).sum().item(),
    }


train_dataset = train_dataset.map(
    count_toks,
    batched=False,
    num_proc=32,
    remove_columns=[
        name
        for name in train_dataset.column_names
        if name not in ["input_ids", "labels"]
    ],
    desc="Tokenizing and reformatting instruction data",
)
tok_count_t = torch.tensor([e["n_toks"] for e in train_dataset], dtype=torch.float32)
pred_tok_count_t = torch.tensor(
    [e["n_pred_toks"] for e in train_dataset], dtype=torch.float32
)

print("*** Input Tokens ***")
print(f"mean toks: {tok_count_t.mean()=}")
print(f"max toks: {tok_count_t.max()=}")
print(f"min toks: {tok_count_t.min()=}")
print(f"std toks: {tok_count_t.std()=}")

print("\n*** Predicted Tokens ***")
print(f"mean toks: {pred_tok_count_t.mean()=}")
print(f"max toks: {pred_tok_count_t.max()=}")
print(f"min toks: {pred_tok_count_t.min()=}")
print(f"std toks: {pred_tok_count_t.std()=}")

print("\n*** Predicted/Input Ratio ***")
print(f"pred toks/input_tok: {pred_tok_count_t.mean()/tok_count_t.mean()=}")

print(f"\n{actual_dataset_len=}")
print(f"Approx total tokens in dataset: {actual_dataset_len * tok_count_t.mean()=}")
print(
    f"Approx total pred tokens in dataset: {actual_dataset_len * pred_tok_count_t.mean()=}"
)
