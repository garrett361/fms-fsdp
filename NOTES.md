# Dev Notes

## DeepSeek-v3

Original parallelism strategy:
- Mesh = (PP, EP + ZeRO-1) = (16, 64). 1024 GPUs
