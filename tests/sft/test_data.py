from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from fms_fsdp.utils.dataloader_utils import CPDataCollator
from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES, encode_sft_example

tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")
tokenizer.chat_template = CHAT_TEMPLATES["tulu"]

print("LOAD")
train_dataset = load_dataset(
    "parquet",
    data_dir="/datasets/long_context_sft/longcontext_121824_cleaned_v1",
    num_proc=32,
)["train"]
train_dataset = train_dataset.select(range(10))

print("MAP")
train_dataset = train_dataset.map(
    partial(encode_sft_example, tokenizer=tokenizer, max_seq_length=2**20),
    batched=False,
    num_proc=32,
    remove_columns=[
        name
        for name in train_dataset.column_names
        if name not in ["input_ids", "labels"]
    ],
    desc="Tokenizing and reformatting instruction data",
)
train_dataset.set_format(type="pt")
train_dataset = train_dataset.filter(lambda example: (example["labels"] != -100).any())

dp_degree = cp_degree = 2

for dp_rank in range(2):
    for cp_rank in range(2):
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dp_degree,
            rank=dp_rank,
            shuffle=True,
            seed=42,
            drop_last=False,
        )
        # Something is wrong w/ the batch sampler
        collate_fn = CPDataCollator(cp_degree=cp_degree, cp_rank=cp_rank)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=1,
        )
        print(f"{cp_rank=}, {dp_rank=}")
        print(next(iter(train_dataloader)))
        print("\n")
