SHELL=/bin/bash
NUM_STEPS=10
REPORT_INTERVAL=$(NUM_STEPS)

all:
	echo "Choose another target"

pp_ep:
	torchrun --nnodes=1 --nproc-per-node=4 main_training_mamba_moe_pp.py \
		--skip_ckpt=True \
		--use_dummy_dataset=True \
		--seq_length=256 \
		--model_variant=deepseek-v2-lite-dev_4_layer \
		--loss_free_balancing_lr=1e-3 \
		--batch_size=2 \
		--num_steps=$(NUM_STEPS) \
		--report_interval=$(REPORT_INTERVAL) \
		--ep_degree=2 \
		--n_microbatches=2 \
		--sharding_strategy="fsdp" \
		--vocab_size=128256 \
		--bos_token=None \
		--eos_token=128000 \
		--logical_shards=1024 \
		--fsdp_activation_checkpointing=True \
		--selective_checkpointing=1 \
		--low_cpu_fsdp=True \
		--use_torch_compile=False



