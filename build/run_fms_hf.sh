#!/bin/sh
# runme - runs exactly one command

# exact command to run:

rm -rf /workspace/tmp && export HF_CACHE=/workspace && export CUDA_VISIBLE_DEVICES=0 && python3 -m tuning.sft_trainer --use_flash_attn=False --model_name_or_path="Maykeye/TinyLLama-v0" --training_data_path=tatsu-lab/alpaca --output_dir=/workspace/tmp --num_train_epochs=5.0 --per_device_train_batch_size=1 --gradient_accumulation_steps=2 --learning_rate=2e-04 --logging_strategy steps --logging_steps 10 --torch_dtype float16 --dataset_text_field="text" --response_template='\n### Response:'

# end (no params, no extra logic)
