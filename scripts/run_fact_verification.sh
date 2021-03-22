#! /bin/bash

set -ex

if [ "$1" != "" ]; then
    python scripts/fact_verification.py \
      --model_name_or_path tals/albert-base-vitaminc-fever \
      --tasks_names vitaminc \
      --data_dir data \
      --do_test \
      --max_seq_length 256 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 128 \
      --learning_rate 2e-5 \
      --max_steps 50000 \
      --save_step 10000 \
      --overwrite_cache \
      --output_dir results/vitaminc_albert_base \
      --test_file $1
else
    python scripts/fact_verification.py \
      --model_name_or_path tals/albert-base-vitaminc-fever \
      --tasks_names vitaminc \
      --data_dir data \
      --do_test \
      --max_seq_length 256 \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 128 \
      --learning_rate 2e-5 \
      --max_steps 50000 \
      --save_step 10000 \
      --overwrite_cache \
      --output_dir results/vitaminc_albert_base \
      --test_tasks vitaminc_real vitaminc_synthetic
fi

  #--fp16 \
  #--test_tasks vitc_real vitc_synthetic \
  #--do_train \
  #--do_predict \
  #--test_on_best_ckpt \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
