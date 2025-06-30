import os
import re
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file
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
from transformers import MambaConfig
from transformers.models.bamba import BambaConfig
from transformers.utils import SAFE_WEIGHTS_NAME


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
        load_path = self._validate_ckp_path(path)
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
                    from transformers import (
                        AutoModelForCausalLM,
                    )

                    hf_model = AutoModelForCausalLM.from_pretrained(hf_ckpt_dir)
                    checkpoint_data = convert_state_dict_to_mamba_ssm(hf_model)[
                        "model_state"
                    ]
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
        if self.rank == 0:
            metadata = kwargs
            metadata["step"] = step
            metadata["model_state"] = model_state
            torch.save(metadata, save_name)
        self.report("Checkpoint written", model_save_time=time.time() - save_time)

        return self._cleanup(), model_state


def convert_ssm_config_to_hf_config(
    config_ssm: Dict,
    **kwargs,
) -> BambaConfig:
    """Convert a config from mamba_ssm to a BambaConfig from here."""
    hf_config: BambaConfig = BambaConfig(**kwargs)

    hf_config.architectures = ["BambaForCausalLM"]

    # Set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_intermediate"]
    hf_config.mamba_n_heads = (
        hf_config.hidden_size * hf_config.mamba_expand
    ) // hf_config.mamba_d_head
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.tie_word_embeddings = config_ssm["tie_embeddings"]

    # currently this script assumes config_ssm belongs to v2
    if config_ssm["ssm_cfg"].get("layer") != "Mamba2":
        raise ValueError("Conversion script only supports Mamba2")

    # Set attention values
    attn_cfg = config_ssm.get("attn_cfg")
    if attn_cfg:
        assert attn_cfg["causal"], "Only support non-causal attention."
        assert not attn_cfg["qkv_proj_bias"], "Only support no qkv bias."
        assert not attn_cfg["out_proj_bias"], "Only support no out bias."
        hf_config.attn_rotary_emb = attn_cfg["rotary_emb_dim"]
        hf_config.num_attention_heads = attn_cfg["num_heads"]
        hf_config.num_key_value_heads = attn_cfg["num_heads_kv"]
        hf_config.rope_theta = attn_cfg["rotary_emb_base"]

    attention_layer_indices = config_ssm.get("attn_layer_idx")
    if attention_layer_indices:
        hf_config.attn_layer_indices = attention_layer_indices

    # Padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def convert_state_dict_to_mamba_ssm(model):
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
        state_dict[f"backbone.layers.{i}.mlp.fc1.weight"] = torch.cat([w1, w2], dim=0)
        if f"model.layers.{i}.self_attn.q_proj.weight" in original_sd:
            q = original_sd.pop(f"model.layers.{i}.self_attn.q_proj.weight")
            k = original_sd.pop(f"model.layers.{i}.self_attn.k_proj.weight")
            v = original_sd.pop(f"model.layers.{i}.self_attn.v_proj.weight")
            state_dict[f"backbone.layers.{i}.mixer.in_proj.weight"] = torch.cat(
                [q, k, v], dim=0
            )
    state_dict["lm_head.weight"] = original_sd.pop("lm_head.weight")
    assert len(original_sd) == 0, original_sd.keys()
    return {"model_state": state_dict}


def save_single_safetensor(
    state_dict: Dict,
    save_directory: str,
    metadata: Dict,
):
    save_file(
        state_dict,
        os.path.join(save_directory, SAFE_WEIGHTS_NAME),
        metadata,
    )


def convert_state_dict_from_mamba_ssm(original_sd: Dict) -> Dict[str, torch.Tensor]:
    state_dict = {}

    for orig_k, param in original_sd.items():
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
            state_dict[k2] = param2
            k = k.replace("mlp.fc1", "feed_forward.up_proj")

        if ("in_proj" in k and orig_k.replace("in_proj", "conv1d") in original_sd) or (
            "out_proj" in k and orig_k.replace("out_proj", "conv1d") in original_sd
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
                state_dict[k2] = param2
                k2 = k.replace("mamba.in_proj", "self_attn.v_proj")
                state_dict[k2] = param3
                k = k.replace("mamba.in_proj", "self_attn.q_proj")

        state_dict[k] = param

    return state_dict


def save_as_single_hf_safetensors_file(
    mamba_cfg: MambaConfig,
    mamba_state_dict: dict[str, torch.Tensor],
    output_dir: str,
    tokenizer,
    precision: str = "fp32",
) -> None:
    token_ids = {}
    for key in [
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ]:
        id = getattr(tokenizer, key, None)
        if id:
            token_ids[key] = id
    tokenizer.save_pretrained(output_dir)

    # there are some configs unsettable by mamba_ssn config, so
    # if there are changes from the defaults, have to pass them into
    # the function
    unsettables = {
        "mamba_d_head": 64,
        "mamba_d_state": 128,
        "mamba_n_groups": 1,
        "rms_norm_eps": 1e-5,
    }

    # Load and save config based on name
    config = asdict(mamba_cfg)

    # convert the config
    hf_config = convert_ssm_config_to_hf_config(
        config_ssm=config,
        **token_ids,
        **unsettables,
    )
    hf_config.save_pretrained(output_dir)

    # FIXME: allow other parameters to pass in
    mamba_state_dict_hf = convert_state_dict_from_mamba_ssm(mamba_state_dict)

    # Save new model to pytorch_dump_path
    dtype = (
        torch.float32
        if precision == "fp32"
        else (torch.bfloat16 if precision == "bf16" else torch.float16)
    )

    save_single_safetensor(
        {k: v.to(dtype) for k, v in mamba_state_dict_hf.items()},
        output_dir,
        metadata={"format": "pt"},
    )
