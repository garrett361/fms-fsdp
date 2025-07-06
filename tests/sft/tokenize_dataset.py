import argparse
import multiprocessing
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer

from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES, encode_sft_example

if __name__ == "__main__":
    cpu_count = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc_load", type=int, default=cpu_count // 2)
    parser.add_argument("--num_proc_map", type=int, default=cpu_count // 2)
    parser.add_argument("--data_dir", type=str, default="/datasets/tokenizers/llama3")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/datasets/long_context_sft/longcontext_121824_cleaned_v1",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.chat_template = CHAT_TEMPLATES["tulu"]
    print("LOAD DATA")
    train_dataset = load_dataset(
        "parquet",
        data_dir=args.data_dir,
        num_proc=args.num_proc_load,
    )["train"]
    print("MAP DATA")
    train_dataset = train_dataset.map(
        partial(
            encode_sft_example,
            tokenizer=tokenizer,
            max_seq_length=None,
        ),
        batched=False,
        num_proc=args.num_proc_map,
        remove_columns=[
            name
            for name in train_dataset.column_names
            if name not in ["input_ids", "labels"]
        ],
        desc="Tokenizing and reformatting instruction data",
    )
