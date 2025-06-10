import pytest
import torch
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from fms_fsdp.utils.dataloader_utils import (
    ChatTokenizerCollator,
    ChatTokenizerCollatorCPCollator,
    get_infinite_iter,
)
from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES

DATA = {
    0: {
        "messages": [
            {"content": "What is the meaning of life?", "role": "user"},
            {"content": "42", "role": "assistant"},
        ]
    },
    1: {
        "messages": [
            {"content": "Why did the chicken cross the road?", "role": "user"},
            {"content": "To get to the other side.", "role": "assistant"},
        ]
    },
    2: {
        "messages": [
            {"content": "Why are we testing these functions?", "role": "user"},
            {"content": "Because it is easy to make mistakes.", "role": "assistant"},
        ]
    },
    3: {
        "messages": [
            {"content": "What is python?", "role": "user"},
            {"content": "A programming language", "role": "assistant"},
        ]
    },
}


TOKENIZER = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")
TOKENIZER.chat_template = CHAT_TEMPLATES["tulu"]
BIG_MAX_SEQ_LEN = 2**30


class Test:
    seed: int = 42

    def test_chat_collator(self) -> None:
        collate_fn = ChatTokenizerCollator(TOKENIZER, BIG_MAX_SEQ_LEN)
        train_dataloader = DataLoader(
            DATA,
            collate_fn=collate_fn,
            batch_size=1,
        )
        data_iter = get_infinite_iter(train_dataloader)
        data = next(data_iter)
        assert isinstance(data, dict)
        assert len(data) == 3
        assert isinstance(data["input_ids"], torch.Tensor)
        # NOTE: @goon - no batch dim, yet
        assert data["input_ids"].ndim == 1
        assert isinstance(data["labels"], torch.Tensor)
        assert data["labels"].ndim == 1
        assert isinstance(data["n_labels_toks"], int)

    def test_chat_and_cp_collator(self) -> None:
        collate_fn = ChatTokenizerCollatorCPCollator(
            TOKENIZER, BIG_MAX_SEQ_LEN, cp_degree=1, cp_rank=0
        )
        train_dataloader = DataLoader(
            DATA,
            collate_fn=collate_fn,
            batch_size=1,
        )
        data_iter = get_infinite_iter(train_dataloader)
        data = next(data_iter)
        assert isinstance(data, dict)
        assert len(data) == 2
        assert isinstance(data["input_ids"], torch.Tensor)
        # NOTE: @goon - batch dim
        assert data["input_ids"].ndim == 2
        assert isinstance(data["labels"], torch.Tensor)
        assert data["labels"].ndim == 2

    @pytest.mark.parametrize("dp_degree", [1, len(DATA)])
    def test_distributed_chat_and_cp_collator(self, dp_degree: int) -> None:
        # Build the non-distributed data to check correctness
        collate_fn = ChatTokenizerCollatorCPCollator(
            TOKENIZER, BIG_MAX_SEQ_LEN, cp_degree=1, cp_rank=0
        )
        # Just using the sampler here to match the shuffling.
        sampler = DistributedSampler(
            DATA,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=self.seed,
            drop_last=False,
        )
        non_dist_train_dataloader = DataLoader(
            DATA,
            sampler=sampler,
            collate_fn=collate_fn,
            batch_size=1,
        )
        # non_dist_data[batch_idx] gives the batch at this idx
        non_dist_data = list(non_dist_train_dataloader)
        # And then the data seen by distributed CP ranks
        cp_degree = len(DATA)

        # dist_data[dp_rank][cp_rank][batch_idx] will give the batch that the indicated rank sees at
        # the indicated batch step
        dist_data = {
            dp_rank: {cp_rank: None for cp_rank in range(cp_degree)}
            for dp_rank in range(dp_degree)
        }
        for dp_rank in range(dp_degree):
            dp_data = []
            for cp_rank in range(cp_degree):
                sampler = DistributedSampler(
                    DATA,
                    num_replicas=dp_degree,
                    rank=dp_rank,
                    shuffle=True,
                    seed=self.seed,
                    drop_last=False,
                )
                collate_fn = ChatTokenizerCollatorCPCollator(
                    TOKENIZER, BIG_MAX_SEQ_LEN, cp_degree=cp_degree, cp_rank=cp_rank
                )
                loader = DataLoader(
                    DATA,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    batch_size=1,
                )
                dist_data[dp_rank][cp_rank] = list(loader)

        # Verify correctness
        batches_per_rank = len(DATA) // dp_degree
        for dp_rank in range(dp_degree):
            for dp_mini_batch_idx in range(batches_per_rank):
                global_batch_idx = dp_rank * batches_per_rank + dp_mini_batch_idx
                for field in ("input_ids", "labels"):
                    expected = non_dist_data[global_batch_idx][field]

                    # Concat data across CP ranks
                    cp_rank_data_list = [
                        d[dp_mini_batch_idx][field] for d in dist_data[dp_rank].values()
                    ]
                    # Every distributed rank should see an even number of tokens, for zig-zag cp
                    # sharding to work
                    assert all(t.numel() % 2 == 0 for t in cp_rank_data_list)
                    seen_concat = torch.cat(cp_rank_data_list, dim=-1)

                    # The dist data may have had padding added to the final rank's data for even
                    # divisibility across CP ranks.
                    n_total_toks = expected.shape[-1]
                    seen_concat_no_padding, padding = (
                        seen_concat[:, :n_total_toks],
                        seen_concat[:, n_total_toks:],
                    )
                    assert torch.all(expected == seen_concat_no_padding)
                    assert torch.all(padding == (0 if field == "input_ids" else -100))

    def test_chat_and_cp_collator_skipping(self) -> None:
        """
        Test that when the labels would all be -100 padding, these examples are skipped.
        """
        collate_fn = ChatTokenizerCollatorCPCollator(
            TOKENIZER, 4, cp_degree=1, cp_rank=0
        )
        train_dataloader = DataLoader(
            DATA,
            collate_fn=collate_fn,
            batch_size=1,
        )
        data_iter = get_infinite_iter(train_dataloader)
        with pytest.raises(RuntimeError, match="trivial None data"):
            next(data_iter)
