from typing import Iterator, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from fms_fsdp.utils.dataset_utils import (
    ArrowHandler,
    AutoHandler,
    BufferDataset,
    CheckpointDataset,
    ParquetHandler,
    PreloadBufferDataset,
    PreprocessDataset,
    SamplingDataset,
    ScalableShardDataset,
    StreamingDocDataset,
    encode_sft_example,
)

_handler_map = {
    "arrow": ArrowHandler,
    "hf_parquet": ParquetHandler,
    "auto": AutoHandler,
}


def causal_lm(data_seq, prompt_len=1):
    """
    Perform causal language modeling by right-shifting the input sequence.
    Sets first prompt_len tokens to be ignored by the loss.
    """
    data_seq = torch.tensor(data_seq, dtype=torch.int)
    t = data_seq.clone()[1:]
    data_seq = data_seq[:-1]
    t[:prompt_len] = -100
    return data_seq, t


def get_dummy_loader(cfg, rank, world_size):
    """
    A simple dummy dataloader yielding incrementing vocab indices in an infinite loop
    """

    class SteadyCounter(torch.utils.data.IterableDataset):
        # Spit out incremental counts of constant length l, modulo vocab size v
        def __init__(self, l, v):
            self.i = 0
            self.l = l
            self.v = v

        def __iter__(self):
            while True:
                out = torch.IntTensor(
                    [x % self.v for x in range(self.i, self.i + self.l)]
                )
                yield out, out
                self.i += self.l

    data = SteadyCounter(cfg.seq_length, cfg.vocab_size)
    return torch.utils.data.DataLoader(data, batch_size=cfg.batch_size)


def get_data_loader(cfg, rank, world_size, dp_degree, postprocess=[causal_lm]):
    """
    Pytorch dataloader for stateful, distributed, and rescalable causal language model (CLM) training.
    Assumes underlying data is sequences of integer values.
    ...
    Args
    ----
    cfg : dataclass
        Training config containing seq len, dataset, dataset weight, datapath, etc. arguments
    rank : int
        Rank of current distributed worker. Used for handling dataset sharding logic.
    world_size : int
        Number of distributed workers. Used for handling dataset sharding logic.
    postprocess : List[Callable]
        Any task-specific postprocessing to apply before handing over data. Steps will apply in
        the order provided by the user. For CLM training, use postprocess=[causal_lm].
    """

    do_cp = False
    if dp_degree != world_size:
        do_cp = True
        cp_worldsize = world_size // dp_degree
        cp_rank = rank % cp_worldsize
        world_size = dp_degree
        rank = rank // cp_worldsize

    datasets, weights, cols = parse_data_args(cfg.datasets, cfg.weights, cfg.col_name)

    # Base streaming dataset. Returns doc chunks in sequence.
    # Implements dataset sampling and rescalability.
    droplist = [
        int(x.strip()) for x in cfg.strip_tokens.split(",") if len(x.strip()) > 0
    ]
    droplist = droplist + [cfg.bos_token, cfg.eos_token, cfg.bol_token, cfg.eol_token]
    assert cfg.file_type in _handler_map, (
        f"File type {cfg.file_type} is not recognized ({list(_handler_map.keys())})"
    )
    if cfg.file_type == "hf_parquet" or cfg.file_type == "auto":
        filehandler = _handler_map[cfg.file_type](cfg.tokenizer_path, cols)
    else:
        filehandler = _handler_map[cfg.file_type](cols)
    # Base reader layer
    data = StreamingDocDataset(
        cfg.data_path,
        rank,
        world_size,
        filehandler,
        cfg.eos_token,
        bos_token=cfg.bos_token,
        strip_tokens=set(droplist),
        min_length=3,
        seed=cfg.seed,
    )
    # Add rescaling/resharding
    data = ScalableShardDataset(
        data,
        cfg.eos_token,
        n_logical_shards=cfg.logical_shards,
    )
    # Add multi-dataset handling
    data = SamplingDataset(
        cfg.data_path,
        data,
        cfg.eos_token,
        datasets=datasets,
        weights=weights,
        verbose=(rank == 0),
    )
    # Wrap above dataset in packing logic to form constant-length lines.
    data = BufferDataset(
        data,
        cfg.seq_length if causal_lm not in postprocess else cfg.seq_length + 1,
        bos_token=cfg.bol_token,
        eos_token=cfg.eol_token,
        pack_hard=True,
    )
    # Shuffle outputs in length 10k buffer. Consecutive lines appear 10k steps apart on average.
    data = PreloadBufferDataset(data, 10000)

    # Apply desired postprocessing steps in sequence
    data = PreprocessDataset(data, torch.IntTensor)
    for p in postprocess:
        data = PreprocessDataset(data, p)

    # Apply CP chunking if using CP
    if do_cp:

        def chunk(x):
            return x[
                (cp_rank * x.size(0)) // cp_worldsize : ((cp_rank + 1) * x.size(0))
                // cp_worldsize
            ]

        data = PreprocessDataset(data, lambda x: (chunk(x[0]), chunk(x[1])))

    # Enable auto-saving
    data = CheckpointDataset(
        data,
        cfg.ckpt_load_path if cfg.resuming_dataset else cfg.ckpt_save_path,
        cfg.checkpoint_interval,
        cfg.batch_size * cfg.grad_acc_steps,
        cfg.ckpt_save_path,
    )
    return torch.utils.data.DataLoader(
        data, num_workers=cfg.num_workers, batch_size=cfg.batch_size
    )


def parse_data_args(datas, weights, cols):
    # Convert csv inputs into corresponding lists of values
    def splitstrip(x):
        if isinstance(x, str):
            return [item.strip() for item in x.split(",")]
        elif isinstance(x, (list, tuple)):
            return list(x)
        elif isinstance(x, (int, float, complex)):
            return [x]
        else:
            raise ValueError(f"arg input {x} cannot be parsed.")

    datas = splitstrip(datas)
    weights = [float(x) for x in splitstrip(weights)]
    cols = splitstrip(cols)
    return datas, weights, cols


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
    should_set_epochs = isinstance(sampler, DistributedSampler)
    while True:
        if should_set_epochs:
            sampler.set_epoch(epoch_idx)
        for item in iter(dataloader):
            if item is not None:
                yield epoch_idx, item
        epoch_idx += 1


class InfiniteCPBatchingIter:
    def __init__(
        self,
        dataloader_list: list[DataLoader],
        weights: list[float],
        max_tokens: int,
        cp_degree: int,
        cp_rank: int,
        pad_id: int = 0,
        separator_id: int = -100,
        seed: int = 42,
        naive_padding_free: bool = False,
    ) -> None:
        self.dataloader_list = dataloader_list
        self.weights = weights
        self.max_tokens = max_tokens
        self.cp_degree = cp_degree
        self.cp_rank = cp_rank
        self.pad_id = pad_id
        self.separator_id = separator_id
        self.seed = seed
        self.naive_padding_free = naive_padding_free
        assert all(w > 0 for w in weights), f"{weights=}"

        self._probs = np.array(self.weights, dtype=np.dtype("float64"))
        self._probs /= self._probs.sum()
        self._generator = np.random.default_rng(self.seed)
        self._infinite_iters = [
            (idx, get_infinite_iter(dl)) for idx, dl in enumerate(self.dataloader_list)
        ]
        self._batch = []
        self._epoch_idxs = [0 for _ in self.dataloader_list]

        self._cp_collator = CPDataCollator(
            cp_degree=cp_degree,
            cp_rank=cp_rank,
            pad_id=pad_id,
            separator_id=separator_id,
        )

    def __iter__(self) -> Iterator[tuple[list[int], dict[str, torch.Tensor]]]:
        while True:
            # Select a dataloader per the given weights
            iter_idx, rand_iter = self._generator.choice(
                self._infinite_iters, p=self._probs
            )
            epoch_idx, item = next(rand_iter)
            self._epoch_idxs[iter_idx] = epoch_idx
            assert isinstance(item, list), f"{item=}"
            assert len(item) == 1, (
                f"Expected batch size 1 inputs, received {len(item)=}"
            )
            input = item[0]["input_ids"]
            if self._batch:
                current_max_tok_example = max(
                    _round_up_to_zig_zag_padding(
                        ex["input_ids"].numel(), self.cp_degree
                    )
                    for ex in self._batch
                )
                tok_in_new_input = _round_up_to_zig_zag_padding(
                    input.numel(), self.cp_degree
                )
                tok_in_batch_with_new_input = (len(self._batch) + 1) * max(
                    current_max_tok_example, tok_in_new_input
                )
                if tok_in_batch_with_new_input > self.max_tokens:
                    self.cp_processed_batch = self._cp_collator(self._batch)
                    yield self._epoch_idxs, self.cp_processed_batch
                    self._batch.clear()
                else:
                    self._batch.extend(item)

            else:
                if input.numel() <= self.max_tokens:
                    self._batch.extend(item)
