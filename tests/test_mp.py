import os
from copy import deepcopy

import torch
from dtest import DTest
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.block import Block
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from fms_fsdp import config
from fms_fsdp.policies import bfSixteen, bfSixteen_fp32buffer
from fms_fsdp.utils.config_utils import get_model_config
from fms_fsdp.utils.train_utils import get_policies


class TestMP(DTest):
    def test(self) -> None:
        # torchrun specific
        cfg = config.train_config()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # some setups
        torch.cuda.set_device(local_rank)
        # get policy
        block = Block
        (
            _,
            wrapping_policy,
            sharding_strategy_policy,
            apply_selective_ac,
            param_init_fn,
        ) = get_policies(cfg, rank, block)

        # get model
        config_data = get_model_config("mamba_9.8b")
        config_data["n_layer"] = 1
        config_data["attn_layer_idx"] = [0]

        mamba_config = MambaConfig(**config_data)
        model = MambaLMHeadModel(mamba_config)
        for buf_name, buf in model.named_buffers():
            self.print_rank0_only(f"{buf_name=}, {buf.dtype=}")

        model_fp32bufs = deepcopy(model)

        rot_emb = model.backbone.layers[0].mixer.rotary_emb
        rot_emb_fp32bufs = model_fp32bufs.backbone.layers[0].mixer.rotary_emb

        # Storage for forward hook inputs and outputs
        hook_data = {
            "rot_emb": {"input": None, "output": None},
            "rot_emb_fp32bufs": {"input": None, "output": None},
        }

        def make_pre_hook(name):
            def hook(module, input):
                # Capture inputs before any in-place modifications
                hook_data[name]["input"] = [t.clone().detach() for t in input]

            return hook

        def make_hook(name):
            def hook(module, input, output):
                hook_data[name]["output"] = [t.clone().detach() for t in output]

            return hook

        rot_emb_pre_hook = rot_emb.register_forward_pre_hook(make_pre_hook("rot_emb"))
        rot_emb_hook = rot_emb.register_forward_hook(make_hook("rot_emb"))
        rot_emb_fp32bufs_pre_hook = rot_emb_fp32bufs.register_forward_pre_hook(
            make_pre_hook("rot_emb_fp32bufs")
        )
        rot_emb_fp32bufs_hook = rot_emb_fp32bufs.register_forward_hook(
            make_hook("rot_emb_fp32bufs")
        )

        fsdp_kwargs = dict(
            auto_wrap_policy=wrapping_policy,
            sharding_strategy=sharding_strategy_policy,
            use_orig_params=True,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            param_init_fn=param_init_fn,
        )
        model = FSDP(model, mixed_precision=bfSixteen, **fsdp_kwargs)
        model_fp32bufs = FSDP(
            model_fp32bufs, mixed_precision=bfSixteen_fp32buffer, **fsdp_kwargs
        )

        self.print_rank0_only("\nbfSixteen Model Buffers:")
        for buf_name, buf in model.named_buffers():
            self.print_rank0_only(f"{buf_name=}, {buf.dtype=}")

        self.print_rank0_only("\nbfSixteen_fp32buffer Model Buffers:")
        for buf_name, buf in model_fp32bufs.named_buffers():
            self.print_rank0_only(f"{buf_name=}, {buf.dtype=}")

        # Run a forward and check again
        bsz, seqlen = 1, 8192
        inputs = torch.randint(128, size=(1, 8192)).cuda()

        with torch.no_grad():
            out = model(inputs).logits
            out_fp32bufs = model_fp32bufs(inputs).logits

        # Verify that the rotary embedding inputs are the same for both modules, but their outputs
        # differ
        self.print_rank0_only("\nTest Inputs")
        for idx, (input_t, input_t_fp32buf) in enumerate(
            zip(hook_data["rot_emb"]["input"], hook_data["rot_emb_fp32bufs"]["input"])
        ):
            self.print_rank0_only(
                f"Input {idx=}: {(input_t_fp32buf - input_t).abs().max()=}"
            )
            self.print_rank0_only(
                f"Input {idx=}: {(input_t_fp32buf - input_t).abs().mean()=}"
            )

        self.print_rank0_only("\nTest Outputs")
        for idx, (output_t, output_t_fp32buf) in enumerate(
            zip(hook_data["rot_emb"]["output"], hook_data["rot_emb_fp32bufs"]["output"])
        ):
            self.print_rank0_only(
                f"Output {idx=}: {(output_t_fp32buf - output_t).abs().max()=}"
            )
            self.print_rank0_only(
                f"Output {idx=}: {(output_t_fp32buf - output_t).abs().mean()=}"
            )
        # NOTE: @goon - I found the discrepancy: with buffer_dtype=torch.bfloat16 the `inv_freq`
        # RoPE tensor ends up being recomputed on GPU, whereas with buffer_dtype=torch.float32
        # (default), `inv_freq` was computed on CPU and then moved to GPU later, and the GPU and CPU
        # computations disagree slightly, resulting in downstream differences.
