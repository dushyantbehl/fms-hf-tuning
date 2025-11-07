#!/bin/sh
# runme - runs exactly one command

# exact command to run:

rm -rf /initializer/tmp && export HF_CACHE=/initializer && export CUDA_VISIBLE_DEVICES=0 && python3 -m tuning.sft_trainer --use_flash_attn=False --model_name_or_path="ibm-granite/granite-4.0-350M" --training_data_path=tatsu-lab/alpaca --output_dir=/initializer/tmp --num_train_epochs=1.0 --per_device_train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=1e-05 --logging_strategy steps --logging_steps 5 --dataset_text_field="text" --max_seq_length 1024 --packing=True --save_model_dir /initializer/tmp --save_strategy epoch --gradient_checkpointing True

# end (no params, no extra logic)
