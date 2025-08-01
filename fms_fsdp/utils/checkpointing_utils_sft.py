import os
import re
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM
from transformers.models.bamba import BambaConfig
from transformers.models.granitemoehybrid import GraniteMoeHybridConfig


def get_latest(targdir, qualifier=lambda x: True, key=os.path.getctime):
    """
    Fetch the full path of the latest file or folder written to target directory,
    subject to name passing the qualifier fn.
    Optional key fn can be used for custom sorting.
    Both functions take full path arguments.
    If directory is empty or nonexistent or no items qualify, return None.
    """
    if os.path.exists(targdir) and len(os.listdir(targdir)) > 0:
        latest = max(
            [
                os.path.join(targdir, x)
                for x in os.listdir(targdir)
                if qualifier(os.path.join(targdir, x))
            ],
            key=key,
        )
        return latest
    return None


def get_oldest(targdir, qualifier=lambda x: True, key=os.path.getctime):
    """
    Fetch the full path of the oldest file or folder written to target directory,
    subject to name passing the qualifier fn.
    Optional key fn can be used for custom sorting.
    Both functions take full path arguments.
    If directory is empty or nonexistent or no items qualify, return None.
    """
    if os.path.exists(targdir) and len(os.listdir(targdir)) > 0:
        oldest = min(
            [
                os.path.join(targdir, x)
                for x in os.listdir(targdir)
                if qualifier(os.path.join(targdir, x))
            ],
            key=key,
        )
        return oldest
    return None


class Checkpointer:
    """
    Manages the checkpoint directory. Saves new checkpoints and deletes old ones after the specified number are written.
    Also handles loading and saving of checkpoints in sharded and unsharded formats.
    Assumes model and optimizer inputs are in FSDP.
    ...
    Args
    ----
    ckpdir : str
        Absolute path to desired save location. Creates a new 'checkpoints/' subfolder at that location.
    n_to_save : int
        Number of volatile checkpoints to maintain at any given time.
    parallel_mode : str
        Write sharded folder ckps (when sharded: 'fsdp' or 'hsdp') or unsharded file ckps (when sharded: 'ddp')
    report_fn : Callable or None
        Optional function for reporting or logging status updates. Expected to handle arbitrary *args, **kwargs.
        Defaults to self._selective_print().
    model_auto_placement : bool
        Optional; If True, auto detect GPU device to move model to, as set in device mesh init

    Methods
    -------
    save : keyword args -> str | None
        Saves dictionary of keyword arg key/value pairs to specified checkpoint directory, deleting old checkpoints
        as necessary. If a checkpoint is deleted, returns the filename of that checkpoint.
    load :
        See docstring for individual function below
    """

    def __init__(
        self,
        ckpdir,
        n_to_save,
        parallel_mode,
        rank,
        local_rank,
        report_fn=None,
        model_auto_placement=False,
    ):
        self.max_ckps = n_to_save
        self.rank = rank
        self.local_rank = local_rank
        self.ckp_path = os.path.join(ckpdir, "checkpoints/")
        os.makedirs(self.ckp_path, exist_ok=True)
        self.p_mode = parallel_mode
        assert parallel_mode in ["fsdp", "hsdp", "ddp"]
        self.report = self._selective_print if report_fn is None else report_fn
        self.model_auto_placement = model_auto_placement

    def _selective_print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args)
            for k, v in kwargs.items():
                print(k, "=", v)

    def _cleanup(self):
        # Clean old checkpoints. Barrier to keep synchronization correct.
        file_to_remove = None
        if (
            self.rank == 0
            and len([x for x in os.listdir(self.ckp_path) if "tmp" in x])
            > self.max_ckps
        ):
            ckp_to_remove = Path(
                get_oldest(self.ckp_path, qualifier=lambda x: "tmp" in x)
            )
            if os.path.isfile(ckp_to_remove):
                ckp_to_remove.unlink()
            else:
                shutil.rmtree(ckp_to_remove)
        return file_to_remove

    def _do_save(self, rank, local_rank):  # , shard_group, replicate_group):
        if self.p_mode == "hsdp":
            return rank == local_rank
        else:
            return True
        # TODO: Distributed writing contingent upon the following fix: https://github.com/pytorch/pytorch/issues/104081
        # if not is_dist:
        #     return (rank == local_rank)
        # else:
        #     a = rank % shard_group.size()
        #     b = rank // shard_group.size()
        #     return True if a % replicate_group.size() == b else False
        # shard_group = model.process_group
        # replicate_group = model.__inter_node_state.process_group

    def _write(self, state_dict, loader_state, process_group, save_name, rank):
        os.makedirs(save_name, exist_ok=True)
        writer = FileSystemWriter(save_name, single_file_per_rank=True)
        if state_dict is not None:
            save_state_dict(
                state_dict=state_dict,
                storage_writer=writer,
                process_group=process_group,
                planner=DefaultSavePlanner(),
            )
        if loader_state is not None:
            loader_state.save_to_path(save_name)

    def _validate_ckp_path(self, path):
        """Interpret path to appropriate checkpoint. If found, return modified path. If not found, return None."""
        # Does path exist and is it non-empty?
        is_file = os.path.isfile(path)
        has_hf_config_json = not is_file and (Path(path) / "config.json").exists()
        if has_hf_config_json:
            return path
        if os.path.exists(path):
            # Is this a file?
            if os.path.isfile(path):
                return path
            # Is this a sharded directory?
            elif "metadata.pth" in os.listdir(path):
                return path
            # Is this a path to a set of checkpoints?
            elif len(os.listdir(path)) > 0:
                latest = get_latest(path)
                if os.path.isfile(latest):
                    return latest
                elif "metadata.pth" in os.listdir(latest):
                    return latest
        return None

    def load(
        self,
        model,
        optimizer,
        dataloader,
        hf_config,
        path="",
        reset_stepcount=False,
        strict=True,
        is_compiled=False,
    ):
        """
        Handle checkpoint loading for model/optimizer/dataloader from given path, according to arguments.
        Defaults to save path for locating an appropriate checkpoint. If a path is provided, will use
        it only if no appropriate checkpoint is found in the save path (in which case it's a job restart).
        Reset_stepcount manually resets optimizer and dataloader states, and stat tracking.
        Strict determines whether to use strict loading or not FOR SINGLEFILE LOADING ONLY.
        Returns model, optimizer, dataloader, current step, and current tokens seen.
        """
        is_resuming = False
        # First check if this is resuming a prior fms run:
        if self._validate_ckp_path(self.ckp_path) is not None:
            path = self.ckp_path
            is_resuming = True
        # Then check the user-supplied path next
        load_path = self._validate_ckp_path(path) or self._validate_ckp_path(
            os.path.join(path, "checkpoints/")
        )
        if load_path is None:
            raise ValueError(
                f"No valid checkpoint detected at {path}; SFT requires a non-trivial starting ckpt."
            )
        else:
            self.report(f"Prior checkpoint {load_path} detected.")
            model_load_time = time.time()
            load_path_obj = Path(load_path)
            if load_path_obj.is_dir() and (load_path_obj / "config.json").exists():
                hf_ckpt_dir = load_path_obj
            elif (
                load_path_obj.is_file()
                and (load_path_obj.parent / "config.json").exists()
            ):
                hf_ckpt_dir = load_path_obj.parent
            else:
                hf_ckpt_dir = None

            if load_path_obj.is_file() or hf_ckpt_dir is not None:
                if hf_ckpt_dir is not None:
                    self.report(f"Loading and converting HF ckpt from {hf_ckpt_dir}.")
                    from transformers import AutoModelForCausalLM

                    hf_model = AutoModelForCausalLM.from_pretrained(hf_ckpt_dir)
                    checkpoint_data = get_fms_state_dict_from_hf_model(
                        hf_config, hf_model
                    )
                    # NOTE: @goon - Open instruct adds a padding token to the tokenizer and adjusts
                    # the vocab size of the embeddings and lm head weights of SFT models. This makes
                    # the vocab larger than that of the fms-fsdp model, due to the addition of
                    # zeros.
                    expected_vocab_size = model.config.vocab_size

                    embedding_key = [k for k in checkpoint_data if "embedding" in k]
                    assert len(embedding_key) == 1, f"{embedding_key=}"
                    embedding_key = embedding_key[0]
                    embedding_weight = checkpoint_data[embedding_key]
                    embedding_vocab_size = embedding_weight.shape[0]

                    lm_head_key = [k for k in checkpoint_data if "lm_head.weight" in k]
                    assert len(lm_head_key) == 1, f"{lm_head_key=}"
                    lm_head_key = lm_head_key[0]
                    lm_head_weight = checkpoint_data[lm_head_key]
                    lm_head_vocab_size = lm_head_weight.shape[0]

                    assert lm_head_vocab_size == embedding_vocab_size, (
                        f"{lm_head_vocab_size=}, {embedding_vocab_size=}"
                    )

                    assert lm_head_vocab_size >= expected_vocab_size, (
                        f"{lm_head_vocab_size=}, {expected_vocab_size=}"
                    )

                    if lm_head_vocab_size > expected_vocab_size:
                        extra_vocab_size = lm_head_vocab_size - expected_vocab_size
                        self.report(
                            f"Pruning {extra_vocab_size} trivial vocab elements from loaded checkpoint."
                        )
                        # Verify the extra entries are zeros
                        lm_head_weight, lm_head_extras = (
                            lm_head_weight[:-extra_vocab_size],
                            lm_head_weight[-extra_vocab_size:],
                        )
                        embedding_weight, embedding_extras = (
                            embedding_weight[:-extra_vocab_size],
                            embedding_weight[-extra_vocab_size:],
                        )
                        # Expect the added embeddings and lm head entries to all be the same
                        # NOTE: @goon - this is apparently failing. Manual inspection shows that the
                        # padding tokens (embedding_extras[0] and lm_head_extras[0]) are getting
                        # some training, while the other extra entries are all the same, as
                        # expected. Unclear why this is happening. TODO: @goon - figure out.
                        self.report(f"{lm_head_extras=}")
                        self.report(f"{embedding_extras=}")
                        embedding_mean_diff = (
                            (
                                embedding_extras
                                - embedding_extras[:1].repeat(
                                    embedding_extras.shape[0], 1
                                )
                            )
                            .abs()
                            .mean()
                        )
                        lm_head_mean_diff = (
                            (
                                lm_head_extras
                                - lm_head_extras[:1].repeat(lm_head_extras.shape[0], 1)
                            )
                            .abs()
                            .mean()
                        )
                        self.report(f"{embedding_mean_diff=}")
                        self.report(f"{lm_head_mean_diff=}")

                        # torch.testing.assert_close(
                        #     embedding_extras,
                        #     embedding_extras[:1].repeat(embedding_extras.shape[0], 1),
                        # )
                        # torch.testing.assert_close(
                        #     lm_head_extras,
                        #     lm_head_extras[:1].repeat(lm_head_extras.shape[0], 1),
                        # )

                        checkpoint_data[lm_head_key] = lm_head_weight
                        checkpoint_data[embedding_key] = embedding_weight

                else:
                    checkpoint_data = torch.load(load_path, map_location="cpu")[
                        "model_state"
                    ]
                if is_compiled:
                    model._orig_mod.load_state_dict(checkpoint_data, strict=strict)
                else:
                    model.load_state_dict(checkpoint_data, strict=strict)
                if self.model_auto_placement:
                    model.to("cuda")
                else:
                    model.to(self.local_rank)
                self.report(
                    f"Checkpoint {load_path} is a single-file checkpoint containing only a model. Optimizer and dataloader are from scratch.",
                    model_load_time=time.time() - model_load_time,
                )
                return model, optimizer, dataloader, 0, 0, 0, is_resuming
            else:
                # Load model
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    state_dict = model.state_dict()
                    model_ckp = {"model_state": state_dict}
                    load_state_dict(
                        state_dict=model_ckp,
                        storage_reader=FileSystemReader(load_path),
                        planner=DefaultLoadPlanner(),
                    )
                    model.load_state_dict(model_ckp["model_state"])
                if self.model_auto_placement:
                    model.to("cuda")
                else:
                    model.to(self.local_rank)
                self.report(model_load_time=time.time() - model_load_time)
                step = 0
                ntok = 0
                n_pred_tok = 0
                # Load metadata
                if is_resuming:
                    metadata = torch.load(os.path.join(load_path, "metadata.pth"))
                    step = metadata.get("step", 0)
                    ntok = metadata.get("tokens_seen", 0)
                    n_pred_tok = metadata.get("pred_tokens_seen", 0)
                    self.report(
                        "Metadata loaded",
                        start_step=step,
                        n_tokens_seen=ntok,
                        n_pred_tokens_seen=n_pred_tok,
                    )
                # Load optimizer
                if optimizer is not None:
                    optim_load_time = time.time()
                    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                        optim_state = load_sharded_optimizer_state_dict(
                            model_state_dict=model.state_dict(),
                            optimizer_key="optimizer_state",
                            storage_reader=FileSystemReader(load_path),
                        )
                    flattened_osd = FSDP.optim_state_dict_to_load(
                        model, optimizer, optim_state["optimizer_state"]
                    )
                    optimizer.load_state_dict(flattened_osd)
                    self.report(optimizer_load_time=time.time() - optim_load_time)
                else:
                    self.report("Skipping optimizer load, no optimizer provided.")
                # Load dataset
                if dataloader is not None:
                    data_load_time = time.time()
                    dataloader.dataset.load_from_path(path)
                    self.report(dataset_load_time=time.time() - data_load_time)
                else:
                    self.report("Skipping dataset load, no dataloader provided.")
                return model, optimizer, dataloader, step, ntok, n_pred_tok, is_resuming

    def save(
        self,
        step,
        model,
        optimizer,
        dataloader,
        **kwargs,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model, optimizer, dataloader)
        rank = self.rank
        save_time = time.time()
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state = model.state_dict()
            optim_state = FSDP.sharded_optim_state_dict(model, optimizer)
        dataloader_state = None if dataloader is None else dataloader.dataset

        save_name = os.path.join(self.ckp_path, "step_" + str(step) + "_ckp")
        state_dict = {"model_state": model_state, "optimizer_state": optim_state}
        if self._do_save(rank, self.local_rank):
            self._write(
                state_dict, dataloader_state, model.process_group, save_name, rank
            )
        else:
            self._write(None, dataloader_state, None, save_name, rank)
        if rank == 0:
            metadata = kwargs
            metadata["step"] = step
            torch.save(metadata, os.path.join(save_name, "metadata.pth"))
        self.report(
            f"Checkpoint saved in {save_name}", model_save_time=time.time() - save_time
        )

        return self._cleanup()

    def save_single_file(
        self,
        step,
        model,
        is_compiled=False,
        return_state_dict_only: bool = False,
        **kwargs,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model)
        pth_path = os.path.join(self.ckp_path[:-12], "pth", "step_" + str(step))
        os.makedirs(pth_path, exist_ok=True)
        save_name = os.path.join(pth_path, "consolidated.00.pth")
        save_time = time.time()
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            if is_compiled:
                model_state = model._orig_mod.state_dict()
            else:
                model_state = model.state_dict()

        if return_state_dict_only:
            return model_state
        if self.rank == 0:
            metadata = kwargs
            metadata["step"] = step
            metadata["model_state"] = model_state
            torch.save(metadata, save_name)
        self.report("Checkpoint written", model_save_time=time.time() - save_time)

        return self._cleanup(), model_state

    def get_full_state_dict(
        self,
        model,
        is_compiled=False,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            if is_compiled:
                model_state = model._orig_mod.state_dict()
            else:
                model_state = model.state_dict()
        return model_state


class FMSHFConvertor(ABC):
    @staticmethod
    @abstractmethod
    def get_fms_state_dict_from_hf_model(
        model: nn.Module,
    ) -> dict[str, torch.Tensor]: ...

    @staticmethod
    @abstractmethod
    def convert_fms_to_hf_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]: ...


class FMSGraniteConvertor(FMSHFConvertor):
    @staticmethod
    def get_fms_state_dict_from_hf_model(
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        original_sd = model.state_dict()
        state_dict = {}

        for orig_k in list(original_sd.keys()):
            k = orig_k.replace("embed_tokens", "embedding")
            k = k.replace("mamba", "mixer")
            k = k.replace("model.norm", "model.norm_f")
            k = re.sub(r"(\d+)\.input_layernorm\.", r"\1.norm.", k)
            k = re.sub(r"(\d+)\.post_attention_layernorm\.", r"\1.norm2.", k)
            k = k.replace("shared_mlp.input_linear", "mlp.fc1")
            k = k.replace("shared_mlp.output_linear", "mlp.fc2")
            k = k.replace("self_attn.o_proj", "mixer.out_proj")
            if k != orig_k:
                state_dict[k.replace("model", "backbone")] = original_sd.pop(orig_k)
        for i in range(len(model.model.layers)):
            if f"model.layers.{i}.self_attn.q_proj.weight" in original_sd:
                q = original_sd.pop(f"model.layers.{i}.self_attn.q_proj.weight")
                k = original_sd.pop(f"model.layers.{i}.self_attn.k_proj.weight")
                v = original_sd.pop(f"model.layers.{i}.self_attn.v_proj.weight")
                state_dict[f"backbone.layers.{i}.mixer.in_proj.weight"] = torch.cat(
                    [q, k, v], dim=0
                )
        state_dict["lm_head.weight"] = original_sd.pop("lm_head.weight")
        assert len(original_sd) == 0, original_sd.keys()
        # [Mamba and HF MLP Differences] Tricky: the MLP code differs between mamba and HF w/r/t how the first linear
        # weights are used. They use different definitions of what chunk forms the gate. Morally:
        # Mamba:
        #     y = self.fc1(x)
        #     y, gate = y.chunk(2, dim=-1)
        #     y = y * self.activation(gate)
        # HF:
        #     y = self.fc1(x)
        #     gate, y = y.chunk(2, dim=-1)
        #     y = y * self.activation(gate)

        # Reorder the weights. If in-place=True, the weights of the original model will be corrupted.
        for k, v in state_dict.items():
            if "mlp.fc1" in k:
                state_dict[k] = torch.cat(list(reversed(v.chunk(2, dim=0))), dim=0)

        return state_dict

    @staticmethod
    def convert_fms_to_hf_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hf_state_dict = {}

        for orig_k, param in state_dict.items():
            k = orig_k.replace("backbone", "model")

            # for embeddings
            k = k.replace("embedding", "embed_tokens")

            # for mixer
            k = k.replace("mixer", "mamba")

            # for final layernorm
            k = k.replace("model.norm_f", "model.norm")

            # for block layernorm
            k = re.sub(r"(\d+)\.norm\.", r"\1.input_layernorm.", k)
            k = re.sub(r"(\d+)\.norm2\.", r"\1.post_attention_layernorm.", k)

            # for mlp
            k = k.replace("mlp.fc1", "shared_mlp.input_linear")
            k = k.replace("mlp.fc2", "shared_mlp.output_linear")

            if (
                "in_proj" in k and orig_k.replace("in_proj", "conv1d") in state_dict
            ) or (
                "out_proj" in k and orig_k.replace("out_proj", "conv1d") in state_dict
            ):
                # then this must be a mamba
                pass
            else:
                # for attn
                # - because mixer was replaced to mamba above
                k = k.replace("mamba.out_proj", "self_attn.o_proj")
                if "mamba.in_proj" in k:
                    m, n = param.shape
                    d = (m - n) // 2
                    param, param2, param3 = torch.split(param, [n, d, d], dim=0)
                    k2 = k.replace("mamba.in_proj", "self_attn.k_proj")
                    hf_state_dict[k2] = param2
                    k2 = k.replace("mamba.in_proj", "self_attn.v_proj")
                    hf_state_dict[k2] = param3
                    k = k.replace("mamba.in_proj", "self_attn.q_proj")

            hf_state_dict[k] = param

        # Reorder the first MLP weights. See [Mamba and HF MLP Differences]
        for k, v in hf_state_dict.items():
            if "shared_mlp.input_linear" in k:
                hf_state_dict[k] = torch.cat(list(reversed(v.chunk(2, dim=0))), dim=0)

        return hf_state_dict


class FMSBambaConvertor(FMSHFConvertor):
    @staticmethod
    def get_fms_state_dict_from_hf_model(
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        original_sd = model.state_dict()
        state_dict = {}

        for orig_k in list(original_sd.keys()):
            # k = orig_k.replace("model", "backbone")
            k = orig_k.replace("embed_tokens", "embedding")
            k = k.replace("mamba", "mixer")
            k = k.replace("final_layernorm", "norm_f")
            k = re.sub(r"(\d+)\.input_layernorm\.", r"\1.norm.", k)
            k = re.sub(r"(\d+)\.pre_ff_layernorm\.", r"\1.norm2.", k)
            k = k.replace("feed_forward.down_proj", "mlp.fc2")
            k = k.replace("self_attn.o_proj", "mixer.out_proj")
            if k != orig_k:
                state_dict[k.replace("model", "backbone")] = original_sd.pop(orig_k)
        for i in range(len(model.model.layers)):
            w1 = original_sd.pop(f"model.layers.{i}.feed_forward.up_proj.weight")
            w2 = original_sd.pop(f"model.layers.{i}.feed_forward.gate_proj.weight")
            state_dict[f"backbone.layers.{i}.mlp.fc1.weight"] = torch.cat(
                [w1, w2], dim=0
            )
            if f"model.layers.{i}.self_attn.q_proj.weight" in original_sd:
                q = original_sd.pop(f"model.layers.{i}.self_attn.q_proj.weight")
                k = original_sd.pop(f"model.layers.{i}.self_attn.k_proj.weight")
                v = original_sd.pop(f"model.layers.{i}.self_attn.v_proj.weight")
                state_dict[f"backbone.layers.{i}.mixer.in_proj.weight"] = torch.cat(
                    [q, k, v], dim=0
                )
        state_dict["lm_head.weight"] = original_sd.pop("lm_head.weight")
        assert len(original_sd) == 0, original_sd.keys()
        return state_dict

    @staticmethod
    def convert_fms_to_hf_state_dict(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hf_state_dict = {}

        for orig_k, param in state_dict.items():
            k = orig_k.replace("backbone", "model")

            # for embeddings
            k = k.replace("embedding", "embed_tokens")

            # for mixer
            k = k.replace("mixer", "mamba")

            # for final layernorm
            k = k.replace("norm_f", "final_layernorm")

            # for block layernorm
            k = re.sub(r"(\d+)\.norm\.", r"\1.input_layernorm.", k)
            k = re.sub(r"(\d+)\.norm2\.", r"\1.pre_ff_layernorm.", k)

            # for mlp
            k = k.replace("mlp.fc2", "feed_forward.down_proj")

            if "mlp.fc1" in k:
                param, param2 = torch.chunk(param, 2, dim=0)
                k2 = k.replace("mlp.fc1", "feed_forward.gate_proj")
                hf_state_dict[k2] = param2
                k = k.replace("mlp.fc1", "feed_forward.up_proj")

            if (
                "in_proj" in k and orig_k.replace("in_proj", "conv1d") in state_dict
            ) or (
                "out_proj" in k and orig_k.replace("out_proj", "conv1d") in state_dict
            ):
                # then this must be a mamba
                pass
            else:
                # for attn
                # - because mixer was replaced to mamba above
                k = k.replace("mamba.out_proj", "self_attn.o_proj")
                if "mamba.in_proj" in k:
                    m, n = param.shape
                    d = (m - n) // 2
                    param, param2, param3 = torch.split(param, [n, d, d], dim=0)
                    k2 = k.replace("mamba.in_proj", "self_attn.k_proj")
                    hf_state_dict[k2] = param2
                    k2 = k.replace("mamba.in_proj", "self_attn.v_proj")
                    hf_state_dict[k2] = param3
                    k = k.replace("mamba.in_proj", "self_attn.q_proj")

            hf_state_dict[k] = param

        return hf_state_dict


def convert_fms_to_hf_state_dict(
    hf_config, fms_state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if isinstance(hf_config, BambaConfig):
        hf_state_dict = FMSBambaConvertor.convert_fms_to_hf_state_dict(fms_state_dict)
    elif isinstance(hf_config, GraniteMoeHybridConfig):
        hf_state_dict = FMSGraniteConvertor.convert_fms_to_hf_state_dict(fms_state_dict)
    else:
        raise TypeError(
            f"{hf_config=} expected to be a BambaConfig or GraniteMoeHybridConfig instance"
        )
    return hf_state_dict


def get_fms_state_dict_from_hf_model(
    hf_config, hf_model: nn.Module
) -> dict[str, torch.Tensor]:
    if isinstance(hf_config, BambaConfig):
        fms_state_dict = FMSBambaConvertor.get_fms_state_dict_from_hf_model(hf_model)
    elif isinstance(hf_config, GraniteMoeHybridConfig):
        fms_state_dict = FMSGraniteConvertor.get_fms_state_dict_from_hf_model(hf_model)
    else:
        raise TypeError(
            f"{hf_config=} expected to be a BambaConfig or GraniteMoeHybridConfig instance"
        )
    return fms_state_dict


def save_hf_model(
    hf_config: BambaConfig | GraniteMoeHybridConfig,
    fms_state_dict: dict[str, torch.Tensor],
    output_dir: str,
    tokenizer,
    precision: str = "fp32",
) -> None:
    hf_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # FIXME: allow other parameters to pass in

    hf_state_dict = convert_fms_to_hf_state_dict(hf_config, fms_state_dict)

    # Save new model to pytorch_dump_path
    dtype = (
        torch.float32
        if precision == "fp32"
        else (torch.bfloat16 if precision == "bf16" else torch.float16)
    )
    hf_model = AutoModelForCausalLM.from_config(hf_config)
    hf_model.load_state_dict(hf_state_dict, strict=True)
    hf_model.to(dtype)
    hf_model.save_pretrained(output_dir, safe_serialization=True)
