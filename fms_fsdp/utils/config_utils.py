from fms.models.llama import LLaMAConfig

from fms_fsdp.config import train_config


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def get_model_config(model_variant):
    if model_variant == "llama2_70b":
        model_config = LLaMAConfig(
            emb_dim=8192,
            multiple_of=4096,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=28672 / 8192,
        )
    elif model_variant == "llama2_34b":
        model_config = LLaMAConfig(
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=48,
            hidden_grow_factor=22016 / 8192,
            max_expected_seq_len=16384,
            rope_theta=1000000.0,
        )
    elif model_variant == "llama2_13b":
        model_config = LLaMAConfig(
            emb_dim=5120,
            nheads=40,
            nlayers=40,
            hidden_grow_factor=13824 / 5120,
        )
    elif model_variant == "llama2_7b":
        model_config = LLaMAConfig(
            hidden_grow_factor=11008 / 4096,
            kvheads=32,
        )
    elif model_variant == "llama2_1.4b":
        model_config = LLaMAConfig(
            emb_dim=2048,
            nheads=16,
            nlayers=24,
            hidden_grow_factor=3,
            kvheads=4,
        )
    elif model_variant == "llama3_8b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_8b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=4096,
            nheads=32,
            kvheads=8,
            nlayers=32,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_1.8b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_1.8b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=2048,
            nheads=16,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_3.2b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_3.2b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=3072,
            nheads=24,
            kvheads=8,
            nlayers=24,
            hidden_grow_factor=8 / 3,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_70b":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=8192,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_70b_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=8192,
            nheads=64,
            kvheads=8,
            nlayers=80,
            hidden_grow_factor=3.5,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "llama3_194m_4k":
        model_config = LLaMAConfig(
            src_vocab_size=128256,
            emb_dim=1024,
            nheads=8,
            nlayers=10,
            max_expected_seq_len=4096,
            rope_theta=500000.0,
        )
    elif model_variant == "mamba_9.8b":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_moe_30b":
        # Somewhat arbitrarily put-together moe model. Loosely based on DSv2-Lite
        # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json
        model_config = {
            "d_model": 4096,
            "d_intermediate": 14336,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
            },
            "moe_layer_idx": list(range(1, 32)),
            "moe_cfg": {
                "n_routed_experts": 64,
                "n_activated_experts": 6,
                "n_shared_experts": 2,
                "hidden_features": 1024,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_moe_lite":
        # Scaled down model for testing. ~2B params.
        model_config = {
            "d_model": 2048,
            "d_intermediate": 5461,
            "n_layer": 16,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [5, 9, 13],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
            },
            "moe_layer_idx": list(range(1, 16)),
            "moe_cfg": {
                "n_routed_experts": 32,
                "n_activated_experts": 4,
                "n_shared_experts": 1,
                "hidden_features": 512,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    # Translated from https://github.com/foundation-model-stack/fms-fsdp/compare/main...moe
    # There they use `d_intermediate` in the expert MLPs, which are GraniteMoeMoE instances.  These
    # classes have two `GraniteMoeParallelExperts` submodules with
    elif model_variant == "mamba_30b_moe":
        model_config = {
            "d_model": 3072,
            "d_intermediate": 1344,
            "n_layer": 32,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 24,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
            },
            "moe_layer_idx": list(range(32)),
            "moe_cfg": {
                "n_routed_experts": 64,
                "n_activated_experts": 8,
                "n_shared_experts": 0,
                "hidden_features": 1344,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_120b_moe":
        model_config = {
            "d_model": 4096,
            "d_intermediate": 896,
            "n_layer": 40,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27, 36],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 32,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
            },
            "moe_layer_idx": list(range(40)),
            "moe_cfg": {
                "n_routed_experts": 256,
                "n_activated_experts": 16,
                "n_shared_experts": 0,
                "hidden_features": 896,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    elif model_variant == "mamba_236b_moe":
        model_config = {
            "d_model": 5120,
            "d_intermediate": 1536,
            "n_layer": 60,
            "vocab_size": 128256,
            "ssm_cfg": {"layer": "Mamba2"},
            "attn_layer_idx": [9, 18, 27, 36, 45, 54],
            "attn_cfg": {
                "causal": True,
                "d_conv": 0,
                "head_dim": 128,
                "num_heads": 40,
                "num_heads_kv": 8,
                "out_proj_bias": False,
                "qkv_proj_bias": False,
                "rotary_emb_dim": 64,
            },
            "moe_layer_idx": list(range(40)),
            "moe_cfg": {
                "n_routed_experts": 160,
                "n_activated_experts": 8,
                "n_shared_experts": 0,
                "hidden_features": 1536,
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False,
        }
    else:
        raise ValueError(f"model variant {model_variant} not supported.")

    return model_config
