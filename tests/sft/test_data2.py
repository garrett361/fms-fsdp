from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from fms_fsdp.utils.dataloader_utils import ChatTokenizerCollator, get_infinite_iter
from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES

tokenizer = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")
tokenizer.chat_template = CHAT_TEMPLATES["tulu"]

print("LOAD")
train_dataset = load_dataset(
    "parquet",
    data_dir="/datasets/long_context_sft/longcontext_121824_cleaned_v1",
    num_proc=32,
)["train"]
train_dataset = train_dataset.select(range(10))
max_seq_length = 2**30


dp_degree = cp_degree = 2
cp_rank = dp_rank = 0

sampler = DistributedSampler(
    train_dataset,
    num_replicas=dp_degree,
    rank=dp_rank,
    shuffle=True,
    seed=42,
    drop_last=False,
)
collate_fn = ChatTokenizerCollator(tokenizer, max_seq_length)
train_dataloader = DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=1,
)
data_iter = get_infinite_iter(train_dataloader)
for batch_idx, item in enumerate(data_iter):
    print(f"{batch_idx=}: {item=}")
