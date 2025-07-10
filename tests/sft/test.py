from functools import partial

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from fms_fsdp.utils.dataloader_utils import (
    ChatTokenizerCollator,
    ChatTokenizerCollatorCPCollator,
    InfiniteCPBatchingIter,
    PretokenizedCollator,
    get_infinite_iter,
)
from fms_fsdp.utils.dataset_utils import CHAT_TEMPLATES, encode_sft_example

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
    4: {
        "messages": [
            {
                "content": "This is a much longer example where I say hello repeatedly: hello hello hello hello hello hello hello.",
                "role": "user",
            },
            {
                "content": "I will say bye repeatedly in return: bye bye bye  bye bye bye bye bye bye bye bye.",
                "role": "assistant",
            },
        ]
    },
    5: {
        "messages": [
            {"content": "Why did the chicken cross the road?", "role": "user"},
            {"content": "None of your business.", "role": "assistant"},
        ]
    },
    6: {
        "messages": [
            {"content": "Why are we testing these functions?", "role": "user"},
            {"content": "Is that a serious question?", "role": "assistant"},
        ]
    },
    7: {
        "messages": [
            {"content": "What is python?", "role": "user"},
            {"content": "A large snake.", "role": "assistant"},
        ]
    },
}


TOKENIZER = AutoTokenizer.from_pretrained("ibm-fms/Bamba-9B")
TOKENIZER.chat_template = CHAT_TEMPLATES["tulu"]
BIG_MAX_SEQ_LEN = 2**30


class Test:
    seed: int = 42

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_chat_collator(self, batch_size: int) -> None:
        collate_fn = ChatTokenizerCollator(TOKENIZER, BIG_MAX_SEQ_LEN)
        train_dataloader = DataLoader(
            DATA,
            collate_fn=collate_fn,
            batch_size=batch_size,
        )
        data_iter = get_infinite_iter(train_dataloader)
        epoch_idx, data = next(data_iter)
        assert isinstance(data, list)
        assert isinstance(data[0], dict)
        for d in data:
            assert len(d) == 3
            assert isinstance(d["input_ids"], torch.Tensor)
            # NOTE: @goon - no batch dim, yet
            assert d["input_ids"].ndim == 1
            assert isinstance(d["labels"], torch.Tensor)
            assert d["labels"].ndim == 1
            assert isinstance(d["n_labels_toks"], int)

    @pytest.mark.parametrize("naive_padding_free", [True, False])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_chat_and_cp_collator(
        self, batch_size: int, naive_padding_free: bool
    ) -> None:
        collate_fn = ChatTokenizerCollatorCPCollator(
            TOKENIZER,
            BIG_MAX_SEQ_LEN,
            cp_degree=1,
            cp_rank=0,
            naive_padding_free=naive_padding_free,
        )
        train_dataloader = DataLoader(
            DATA,
            collate_fn=collate_fn,
            batch_size=batch_size,
        )
        data_iter = get_infinite_iter(train_dataloader)
        epoch_idx, data = next(data_iter)
        assert isinstance(data, dict)
        assert len(data) == 2
        for t in data.values():
            assert isinstance(t, torch.Tensor)
            if naive_padding_free:
                assert t.shape[0] == 1
            else:
                assert t.shape[0] == batch_size

    @pytest.mark.parametrize("dp_degree", [1, len(DATA)])
    @pytest.mark.parametrize("cp_degree", [1, len(DATA)])
    def test_distributed_chat_and_cp_collator(
        self, dp_degree: int, cp_degree: int
    ) -> None:
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

        # dist_data[dp_rank][cp_rank][batch_idx] will give the batch that the indicated rank sees at
        # the indicated batch step
        dist_data = {
            dp_rank: {cp_rank: None for cp_rank in range(cp_degree)}
            for dp_rank in range(dp_degree)
        }
        for dp_rank in range(dp_degree):
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

    @pytest.mark.parametrize("naive_padding_free", [True, False])
    @pytest.mark.parametrize("cp_degree", [1, 2, 4])
    @pytest.mark.parametrize("num_datasets", [1, 2, 3])
    def test_infinite_cp_batching_iter(
        self, cp_degree: int, num_datasets: int, naive_padding_free: bool
    ) -> None:
        weights = [float(n) for n in range(1, num_datasets + 1)]
        pretok_dataset = Dataset.from_list(list(DATA.values()))
        pretok_dataset = pretok_dataset.map(
            partial(
                encode_sft_example,
                tokenizer=TOKENIZER,
                max_seq_length=None,
            ),
            batched=False,
            remove_columns=[
                name
                for name in pretok_dataset.column_names
                if name not in ["input_ids", "labels"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )

        # First without any CP complications
        dataloader_list = []
        for idx in range(num_datasets):
            # Make the different datasets different sizes.
            sliced_pretok_dataset = pretok_dataset.select(
                range(len(pretok_dataset) - idx)
            )
            sampler = DistributedSampler(
                sliced_pretok_dataset,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=self.seed,
                drop_last=False,
            )
            train_dataloader = DataLoader(
                sliced_pretok_dataset,
                sampler=sampler,
                collate_fn=PretokenizedCollator(),
                batch_size=1,
            )
            dataloader_list.append(train_dataloader)
        max_tokens = 100
        max_reps = 10
        data_iter = InfiniteCPBatchingIter(
            dataloader_list=dataloader_list,
            weights=weights,
            max_tokens=max_tokens,
            cp_degree=1,
            cp_rank=0,
            naive_padding_free=naive_padding_free,
        )
        non_cp_batches = []
        for rep_idx, (_, batch) in enumerate(data_iter):
            if rep_idx > max_reps:
                break
            non_cp_batches.append(batch)
            input, label = batch["input_ids"], batch["labels"]
            assert input.numel() == label.numel()
            assert input.numel() <= max_tokens, f"{input.numel()=}, {max_tokens=}"

        # And then CP
        cp_data_iters = [None] * cp_degree
        for cp_rank in range(cp_degree):
            sliced_pretok_dataset = pretok_dataset.select(
                range(len(pretok_dataset) - idx)
            )
            sampler = DistributedSampler(
                sliced_pretok_dataset,
                num_replicas=1,
                rank=0,
                shuffle=True,
                seed=self.seed,
                drop_last=False,
            )
            train_dataloader = DataLoader(
                sliced_pretok_dataset,
                sampler=sampler,
                collate_fn=PretokenizedCollator(),
                batch_size=1,
            )
            cp_data_iters[cp_rank] = data_iter = InfiniteCPBatchingIter(
                dataloader_list=dataloader_list,
                weights=weights,
                max_tokens=max_tokens,
                cp_degree=cp_degree,
                cp_rank=cp_rank,
                naive_padding_free=naive_padding_free,
            )
        for batch, cp_batch_tuple in zip(non_cp_batches, zip(*cp_data_iters)):
            inputs, labels = batch["input_ids"], batch["labels"]

            cp_batches = [b[1] for b in cp_batch_tuple]
            cp_inputs = torch.cat([b["input_ids"] for b in cp_batches], dim=-1)
            cp_labels = torch.cat([b["labels"] for b in cp_batches], dim=-1)

            if naive_padding_free:
                assert input.shape[0] == 1
                assert labels.shape[0] == 1
                assert cp_inputs.shape[0] == 1
                assert cp_labels.shape[0] == 1

            assert cp_inputs.numel() == cp_labels.numel()
            assert cp_inputs.numel() <= max_tokens, (
                f"{cp_inputs.numel()=}, {max_tokens=}"
            )

            # Should agree up to possible CP padding differences
            num_padding_elements = (inputs == 0).sum(dim=-1).min()
            if num_padding_elements:
                inputs = inputs[:, :-num_padding_elements]
                labels = labels[:, :-num_padding_elements]
            num_cp_padding_elements = (cp_inputs == 0).sum(dim=-1).min()
            if num_cp_padding_elements:
                cp_inputs = cp_inputs[:, :-num_cp_padding_elements]
                cp_labels = cp_labels[:, :-num_cp_padding_elements]

            torch.testing.assert_close(inputs, cp_inputs)
            torch.testing.assert_close(labels, cp_labels)
