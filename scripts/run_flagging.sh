#! /bin/bash

set -ex

python scripts/factual_flagging.py \
  --model_name_or_path tals/albert-base-vitaminc_flagging \
  --data_dir data \
  --input_features raw_sents \
  --do_test \
  --max_seq_length 256 \
  --fp16 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 128 \
  --learning_rate 2e-5 \
  --max_steps 12000 \
  --save_step 4000 \
  --overwrite_cache \
  --output_dir results/vitaminc_flagging

  #--do_train \
  #--model_name_or_path albert-base-v2 \
  #--do_eval \
  #--eval_all_checkpoints \
  #--test_on_best_ckpt \
  #--do_predict \
