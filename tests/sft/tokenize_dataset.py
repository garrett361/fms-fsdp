from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer

from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES, encode_sft_example

tokenizer = AutoTokenizer.from_pretrained("/datasets/tokenizers/llama3")
tokenizer.chat_template = CHAT_TEMPLATES["tulu"]
train_dataset = load_dataset(
    "parquet",
    data_dir="/datasets/long_context_sft/longcontext_121824_cleaned_v1",
    num_proc=4,
)["train"]
train_dataset = train_dataset.map(
    partial(
        encode_sft_example,
        tokenizer=tokenizer,
        max_seq_length=None,
    ),
    batched=False,
    num_proc=48,
    remove_columns=[
        name
        for name in train_dataset.column_names
        if name not in ["input_ids", "labels"]
    ],
    desc="Tokenizing and reformatting instruction data",
)
