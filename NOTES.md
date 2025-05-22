# Dev Notes

## Best Practices/Goals

- Avoid activation checkpointing for the routed experts. Otherwise, incurs additional all-to-all,
which are the main bottleneck.


## DeepSeek-v3

Original parallelism strategy:
- Best guess at mesh = (ZeRO-1, PP, EP + ZeRO-???) = (2, 16, 64). 2048 GPUs


## Typical Memory Costs

- Embed + LM Head: `~= 4 GiB * (vocab_size / 128256) * (d_model / 4096) * (B_per_param / 4)`
- Routed experts, per expert `~= 0.15 GiB * (d_intermediate / 2048) * (d_model / 7168) * (B_per_param / 4)`
- Attention: `~= 0.5GiB (d_model / 4096) * (head_dim / 128) * ((n_heads + n_kv_heads) / 256)* (B_per_param / 4)`


# Non-OOM Configs

Minimal cfgs which avoid OOMs on A100s. Not at all optimized.

| Model | EP | PP| BS | SEQ_LEN| OTHER|
| ----- | ----- | -----| ----- | -----| -----|
| `deepseek-v2-lite` | 16 | 1| 1 | 4096| `no_reshard=False`|
| `deepseek-v2` | ??? | ???| ??? | ???| ???|
| `deepseek-v3` | ??? | ???| ??? | ???| ???|
| `llama-4-maverick` | ??? | ???| ??? | ???| ???|


- `deepseek-v2-lite`

