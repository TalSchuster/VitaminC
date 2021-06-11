#! /bin/bash

set -ex

python scripts/rationale.py \
       --model_name_or_path tals/albert-base-vitaminc_rationale \
       --data_dir data \
       --max_steps 20000 \
       --save_steps 5000 \
       --per_device_train_batch_size 32 \
       --per_device_eval_batch_size 128 \
       --learning_rate 2e-5 \
       --fp16 \
       --do_test \
       --output_dir results/rationale \
       --sparsity_weight 0.2 \
       --supervised_weight 1 \
       --use_pretrained_classifier \
       --pretrained_classifier_path tals/albert-base-vitaminc_wnei-fever \
       --fix_pretrained_classifier \
       "$@"

       #--overwrite_cache \
       #--do_eval \
       #--do_train \
       #--model_name_or_path albert-base-v2 \
