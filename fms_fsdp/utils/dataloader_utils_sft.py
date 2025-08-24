from dataclasses import dataclass, field
from warnings import warn
import pathlib
from typing import Iterator, Optional, Union, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

### From open-instruct

# Chat templates
# flake8: noqa
# note we added `{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}`
# because we want the template to not output eos_token if `add_generation_prompt=True`
CHAT_TEMPLATES = {
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "granite": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n' + message['content'] + eos_token }}"
        "{% endif %}"
        "{% else %}"
        "{{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "granite2": (
        "{%- if messages[0]['role'] == 'system' %}"
        "{%- set system_message = messages[0]['content'] %}"
        "{%- set loop_messages = messages[1:] %}"
        "{%- else %}"
        "{%- set system_message = '' %}"
        "{%- set loop_messages = messages %}"
        "{%- endif %}"
        "{%- if system_message|length > 0 %}"
        "{{ '<|start_of_role|>system<|end_of_role|>' + system_message + eos_token + '\n' }}"
        "{%- endif %}"
        "{%- if tools %}"
        "{{ '<|start_of_role|>tools<|end_of_role|>' }}"
        "{{ tools | tojson(indent=4) }}"
        "{{ eos_token + '\n' }}"
        "{%- endif %}"
        "{%- if documents %}"
        "{{ '<|start_of_role|>documents<|end_of_role|>' }}"
        "{%- for document in documents %}"
        "{{ 'Document ' + loop.index0|string + '\n' }}"
        "{{ document['text'] }}"
        "{%- if not loop.last %}"
        "{{ '\n\n' }}"
        "{%- endif %}"
        "{%- endfor %}"
        "{{ eos_token + '\n' }}"
        "{%- endif %}"
        "{%- for message in loop_messages %}"
        "{{ '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + eos_token + '\n' }}"
        "{%- if loop.last and add_generation_prompt %}"
        "{{ '<|start_of_role|>assistant' }}"
        "{%- if controls %}"
        "{{ ' ' + controls|tojson() }}"
        "{%- endif %}"
        "{{ '<|end_of_role|>' }}"
        "{%- endif %}"
        "{%- endfor %}"
    ),
}
# flake8: noqa


def get_chat_template(name_or_path: str) -> str:
    if name_or_path in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[name_or_path]
    path = pathlib.Path(name_or_path)
    if path.exists():
        with open(path) as f:
            chat_template = f.read()
        return chat_template
    else:
        raise ValueError(
            f"{name_or_path} is unknown and is not a path to a readable template file."
        )


def encode_sft_example(example, tokenizer, max_seq_length):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[
                        :message_idx
                    ],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "n_labels_toks": (labels != -100).sum().item(),
        # "attention_mask": attention_mask.flatten(),
    }


class ChatTokenizerCollator:
    """
    Takes in raw text and encodes with the chat template.
    """

    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, examples: list[str]):
        out = [
            encode_sft_example(
                e, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length
            )
            for e in examples
        ]

        # Using None to signify not enough non-trivial tokens.
        # NOTE: @goon - : not ideal for batch size > 1 # since we are throwing away the whole batch
        # if even 1 example is trivial. Don't think this will matter hugely in practice.
        if any(o["n_labels_toks"] == 0 for o in out):
            return None
        return out


def _round_up_to_zig_zag_padding(num_toks: int, cp_degree: int) -> int:
    return 2 * cp_degree * ((num_toks + 2 * cp_degree - 1) // (2 * cp_degree))


class CPDataCollator:
    def __init__(
        self,
        cp_degree: int,
        cp_rank: int,
        pad_id: int = 0,
        separator_id: int = -100,
        naive_padding_free: bool = False,
    ):
        self.cp_degree = cp_degree
        self.cp_rank = cp_rank
        self.pad_id = pad_id
        self.separator_id = separator_id
        self.naive_padding_free = naive_padding_free

    def __call__(self, features: list[dict[str, torch.Tensor]]):
        """
        Return None if there are non non-trivial preds
        """
        if features is None:
            # Handling the None cases from ChatTokenizerCollator
            return None

        # features is a list[dict[str, Union[list[int], Tensor]]], make it always be list[dict[str,
        # Tensor]]
        if not torch.is_tensor(features[0]["input_ids"]):
            features = [
                {
                    "input_ids": torch.tensor(f["input_ids"]),
                    "labels": torch.tensor(f["labels"]),
                }
                for f in features
            ]

        if self.naive_padding_free:
            return self._collate_with_naive_padding_free(features)
        return self._collate_with_padding(features)

    def _collate_with_padding(self, features) -> dict[str, torch.Tensor]:
        ret = {"input_ids": [], "labels": []}
        # Need padding, both to align all elements in the batch and also to meet CP divisibility
        # requirements
        seqlens = [f["input_ids"].numel() for f in features]
        max_seqlen = max(seqlens)

        # NOTE: @goon - if using zig-zag, the per-rank tok counts also need to be even, hence
        # the factors of two
        padded_numel = _round_up_to_zig_zag_padding(max_seqlen, self.cp_degree)
        for item in features:
            input_ids = item["input_ids"]
            labels = item["labels"]
            # [CP causal shifting]
            # At this point, the input and labels are in exact causal alignment. We want to shift
            # the label indices over so that input_idx[t] is the input at time step t, while
            # labels[t] is the ground-truth tok at time t + 1. Then the loss is computed like
            # (schematically):
            #
            # ```py
            # out = model(inputs_ids)
            # loss = F.cross_entropy(out, labels)
            # ```

            # Shift and mask the final token
            labels = labels.roll(-1)
            labels[-1] = self.separator_id

            n_pad_toks = padded_numel - input_ids.numel()
            if n_pad_toks > 0:
                input_ids_padding = torch.full(
                    (n_pad_toks,),
                    self.pad_id,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                labels_padding = torch.full(
                    (n_pad_toks,),
                    self.separator_id,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                input_ids = torch.cat([input_ids, input_ids_padding])
                labels = torch.cat([labels, labels_padding])

            # Chunk up and divide among ranks
            ret["input_ids"].append(
                torch.chunk(input_ids, chunks=self.cp_degree, dim=-1)[self.cp_rank]
            )
            ret["labels"].append(
                torch.chunk(labels, chunks=self.cp_degree, dim=-1)[self.cp_rank]
            )

        # Stack and add a batch dimension
        ret["input_ids"] = torch.stack(ret["input_ids"], dim=0)
        ret["labels"] = torch.stack(ret["labels"], dim=0)
        return ret

    def _collate_with_naive_padding_free(self, features) -> dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        for item in features:
            input_ids = item["input_ids"]
            labels = item["labels"]
            # Shift and mask the final token
            labels = labels.roll(-1)
            labels[-1] = self.separator_id
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Concatenate, pad, and chunk
        n_toks = sum(i.numel() for i in input_ids_list)
        padded_numel = _round_up_to_zig_zag_padding(n_toks, self.cp_degree)
        n_pad_toks = padded_numel - n_toks
        if n_pad_toks > 0:
            input_ids_padding = torch.full(
                (n_pad_toks,),
                self.pad_id,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels_padding = torch.full(
                (n_pad_toks,),
                self.separator_id,
                device=labels.device,
                dtype=labels.dtype,
            )
            input_ids_list.append(input_ids_padding)
            labels_list.append(labels_padding)

        # Concatenate, add a batch dimension, and chunk:
        input_ids = torch.cat(input_ids_list, dim=-1)[None]
        input_ids = input_ids.chunk(chunks=self.cp_degree, dim=-1)[self.cp_rank]

        labels = torch.cat(labels_list, dim=-1)[None]
        labels = labels.chunk(chunks=self.cp_degree, dim=-1)[self.cp_rank]

        return {"input_ids": input_ids, "labels": labels}


class ChatTokenizerCollatorCPCollator:
    """
    For tokenizing on the fly.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int,
        cp_degree: int,
        cp_rank: int,
        pad_id: int = 0,
        separator_id: int = -100,
        naive_padding_free: bool = False,
    ):
        self.cp_degree = cp_degree
        self.cp_rank = cp_rank
        self.separator_id = separator_id
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.naive_padding_free = naive_padding_free

        self.chat_collator = ChatTokenizerCollator(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        self.cp_collator = CPDataCollator(
            cp_degree=cp_degree,
            cp_rank=cp_rank,
            pad_id=pad_id,
            separator_id=separator_id,
            naive_padding_free=naive_padding_free,
        )

    def __call__(self, example):
        return self.cp_collator(self.chat_collator(example))


class PretokenizedCollator:
    """
    For handling pre-tokenized data
    """

    def __call__(self, examples: list[dict[str, Union[list[int], int]]]):
        out = []
        for ex in examples:
            item = {}
            for k, v in ex.items():
                if torch.torch.is_tensor(v):
                    item[k] = v
                elif isinstance(v, list):
                    item[k] = torch.tensor(v)
            out.append(item)

        return out


def get_infinite_iter(dataloader: DataLoader):
    """
    Infinite iterator, skipping over the None cases above.
    """
    epoch_idx = 0
    sampler = dataloader.sampler
    should_set_epoch = isinstance(sampler, DistributedSampler)
    while True:
        if should_set_epoch:
            sampler.set_epoch(epoch_idx)
        for item in iter(dataloader):
            if item is not None:
                yield epoch_idx, item
        epoch_idx += 1


@dataclass
class DatasetStats:
    epoch_idx: list[int] = field(default_factory=list)
    examples_seen: list[int] = field(default_factory=list)
    tokens_seen: list[int] = field(default_factory=list)
    pred_tokens_seen: list[int] = field(default_factory=list)


class InfiniteCPBatchingIter:
    """
    Inifinite data iterator.

    If max_out_tokens=True, this calss greedily packs full examples from `dataloader_list`, grouping
    up to max_tokens = batch_size * seq_length tokens in a batch. Examples are drawn per-dataset
    according to `weights`examples are split for context-parallel training. If
    `naive_padding_free=True`, the examples are all concatenated together, otherwise they are
    batched and padded. The iterator returns a tuple of:
    0) A DatatsetStats instance
    1) The number of examples packed in the batch
    2) The cp-processed batch, a dict[str, Tensor] with `input_ids`, and `labels` keys in HF style.
       This iterator also performs the causal shifting of the labels.
    """

    def __init__(
        self,
        dataloader_list: list[DataLoader],
        weights: list[float],
        batch_size: int,
        seq_length: int,
        max_out_tokens: bool,
        cp_degree: int,
        cp_rank: int,
        pad_id: int = 0,
        separator_id: int = -100,
        seed: int = 42,
        naive_padding_free: bool = False,
        weight_by: Literal["example", "token", "pred_token"] = "example",
    ) -> None:
        if not all(w > 0 for w in weights):
            raise ValueError(f"{weights=} must all be strictly positive")
        if weight_by not in ["example", "token", "pred_token"]:
            raise ValueError(
                f"{weight_by=} must be one of 'example', 'token', or 'pred_token'"
            )
        self.dataloader_list = dataloader_list
        self.weights = torch.tensor(weights, dtype=torch.float32)
        # Normalize:
        self.weights /= self.weights.sum()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_out_tokens = max_out_tokens
        self.max_tokens = batch_size * seq_length
        self.cp_degree = cp_degree
        self.cp_rank = cp_rank
        self.pad_id = pad_id
        self.separator_id = separator_id
        self.seed = seed
        self.naive_padding_free = naive_padding_free
        self.weight_by = weight_by

        self._stats = DatasetStats(
            epoch_idx=torch.zeros(len(dataloader_list), dtype=torch.int64),
            examples_seen=torch.zeros(len(dataloader_list), dtype=torch.int64),
            tokens_seen=torch.zeros(len(dataloader_list), dtype=torch.int64),
            pred_tokens_seen=torch.zeros(len(dataloader_list), dtype=torch.int64),
        )
        self._infinite_iters = [get_infinite_iter(dl) for dl in self.dataloader_list]
        self._batch = []

        self._cp_collator = CPDataCollator(
            cp_degree=cp_degree,
            cp_rank=cp_rank,
            pad_id=pad_id,
            separator_id=separator_id,
            naive_padding_free=naive_padding_free,
        )

    def __iter__(self) -> Iterator[tuple[DatasetStats, int, dict[str, torch.Tensor]]]:
        return self

    def __next__(self) -> tuple[DatasetStats, int, dict[str, torch.Tensor]]:
        while True:
            # Select a dataloader per the given weights.
            if self.weight_by == "example":
                iter_idx = torch.multinomial(self.weights, 1).item()
            elif self.weight_by == "token":
                # Choose the most under-represented dataset by total token.
                expected_tokens = self._stats.tokens_seen.sum() * self.weights
                diff_tokens = self._stats.tokens_seen - expected_tokens
                iter_idx = diff_tokens.argmin().item()
            elif self.weight_by == "pred_token":
                # Choose the most under-represented dataset by total pred token.
                expected_pred_tokens = self._stats.pred_tokens_seen.sum() * self.weights
                diff_pred_tokens = self._stats.pred_tokens_seen - expected_pred_tokens
                iter_idx = diff_pred_tokens.argmin().item()
            rand_iter = self._infinite_iters[iter_idx]
            epoch_idx, item = next(rand_iter)
            assert isinstance(item, list), f"{item=}"
            assert len(item) == 1, (
                f"Expected batch size 1 inputs, received {len(item)=}"
            )
            n_tok_next_item = item[0]["input_ids"].numel()
            n_pred_tok_next_item = (item[0]["labels"] != self.separator_id).sum()
            if n_tok_next_item > self.max_tokens:
                if not self.cp_rank:
                    warn(
                        f"Skipping data example with {n_tok_next_item} tokens > {self.max_tokens=} ",
                        stacklevel=1,
                    )
                continue
            self._stats.epoch_idx[iter_idx] = epoch_idx
            self._stats.examples_seen[iter_idx] += 1
            self._stats.tokens_seen[iter_idx] += n_tok_next_item
            self._stats.pred_tokens_seen[iter_idx] += n_pred_tok_next_item
            if not self._should_yield_batch(n_tok_next_item):
                self._batch.extend(item)
            else:
                self.cp_processed_batch = self._cp_collator(self._batch)
                batch_size = len(self._batch)
                self._batch.clear()
                self._batch.extend(item)
                return self._stats, batch_size, self.cp_processed_batch

    def _should_yield_batch(self, n_tok_next_item: int) -> bool:
        if not self._batch:
            return False

        if not self.max_out_tokens:
            return len(self._batch) == self.batch_size

        if self.naive_padding_free:
            current_tok_in_batch = sum(b["input_ids"].numel() for b in self._batch)
            tok_in_batch_with_new_input = _round_up_to_zig_zag_padding(
                current_tok_in_batch + n_tok_next_item, self.cp_degree
            )
        else:
            current_max_tok_example = max(
                _round_up_to_zig_zag_padding(ex["input_ids"].numel(), self.cp_degree)
                for ex in self._batch
            )
            tok_in_new_input = _round_up_to_zig_zag_padding(
                n_tok_next_item, self.cp_degree
            )
            tok_in_batch_with_new_input = (len(self._batch) + 1) * max(
                current_max_tok_example, tok_in_new_input
            )
        return tok_in_batch_with_new_input > self.max_tokens
