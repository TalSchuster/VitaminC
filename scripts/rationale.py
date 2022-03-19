# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for token-based rationales."""

import logging
import os
import sys
import functools
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import HfArgumentParser
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed

from vitaminc.processing.rationale import RationaleDataset
from vitaminc.modeling.rationale import (
        AlbertForTokenRationale,
        compute_metrics_fn,
        )

logger = logging.getLogger(__name__)

# Dataset name to path.
NAME_TO_DATA_DIR = {
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    num_labels: Optional[int] = field(
        default=3, metadata={"help": "Number of labels (NEI/SUPPORTS/REFUTES)."}
    )
    use_pretrained_classifier: Optional[bool] = field(
        default=False, metadata={"help": "Use a pretrained classifier and keep params fixed."}
    )
    pretrained_classifier_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained classifier model."}
    )
    fix_pretrained_classifier: Optional[bool] = field(
        default=False, metadata={"help": "Do not update classifier parameters."}
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help": "Gumbel Softmax temperature."}
    )
    confusion_weight: Optional[float] = field(
        default=1.0, metadata={"help": "Lambda for loss margin using masked evidence.."}
    )
    continuity_weight: Optional[float] = field(
        default=0, metadata={"help": "Lambda for continuity penalty."}
    )
    sparsity_weight: Optional[float] = field(
        default=0.1, metadata={"help": "Lambda for sparsity penalty."}
    )
    supervised_weight: Optional[float] = field(
        default=1.0, metadata={"help": "Lambda for supervised masking."}
    )
    unsupervised_weight: Optional[float] = field(
        default=0.3, metadata={"help": "Lambda for examples without supervision."}
    )
    eval_all_checkpoints: Optional[bool] = field(
        default=False, metadata={"help": "Run evaluation on all checkpoints."}
    )
    do_test: Optional[bool] = field(
        default=False, metadata={"help": "Run evaluation on test set (needs labels)."}
    )
    eval_on_datasets: Optional[List[str]] = field(
        default=None, metadata={"help": "List of additional datasets to test on."}
    )
    top_k: Optional[int] = field(
        default=-1, metadata={"help": "Take top-k masked tokens."}
    )
    gen_on_datasets: Optional[List[str]] = field(
        default=None, metadata={"help": "Generate masked data."}
    )


@dataclass
class DataTrainingArguments:
    """
    Data arguments.
    """
    data_dir: str = field(
        default="data/",
        metadata={"help": "The input data dir with jsonl files."}
    )
    data_cache_dir: str = field(
        default="data/cached",
        metadata={"help": "The cache dir for data."}
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    os.makedirs(training_args.output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(training_args.output_dir, 'log.txt'))
    logging.getLogger("transformers").setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logging.getLogger("transformers").addHandler(fh)
    logging.root.addHandler(fh)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Create tokenizer/model.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,  # Our preprocessing is not supported by the fast version.
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=model_args.num_labels,
        cache_dir=model_args.cache_dir,
    )

    # Special args.
    config.nei_index = 2
    config.confusion_weight = model_args.confusion_weight
    config.temperature = model_args.temperature
    config.continuity_weight = model_args.continuity_weight
    config.sparsity_weight = model_args.sparsity_weight
    config.supervised_weight = model_args.supervised_weight
    config.mask_token_id = tokenizer.mask_token_id
    config.use_pretrained_classifier = model_args.use_pretrained_classifier
    config.pretrained_classifier_path = model_args.pretrained_classifier_path
    config.fix_pretrained_classifier = model_args.fix_pretrained_classifier
    config.unsupervised_weight = model_args.unsupervised_weight
    config.top_k = model_args.top_k
    top_k = model_args.top_k

    model = AlbertForTokenRationale.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get our special datasets.
    train_dataset = (
        RationaleDataset(data_args, tokenizer=tokenizer, mode="train", cache_dir=data_args.data_cache_dir)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        RationaleDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=data_args.data_cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        RationaleDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=data_args.data_cache_dir)
        if (model_args.do_test or training_args.do_predict)
        else None
    )

    # Get extra datasets for testing.
    if model_args.eval_on_datasets:
        extra_test_datasets = {}
        for key in model_args.eval_on_datasets:
            data_args.data_dir = NAME_TO_DATA_DIR[key]
            extra_test_datasets[key] = RationaleDataset(
                data_args, tokenizer=tokenizer, task_name=key, mode="test", cache_dir=data_args.data_cache_dir)
    else:
        extra_test_datasets = {}

    # Get extra datasets for generation.
    if model_args.gen_on_datasets:
        extra_gen_datasets = {}
        for key in model_args.gen_on_datasets:
            data_args.data_dir = NAME_TO_DATA_DIR[key]
            extra_gen_datasets[key] = RationaleDataset(
                data_args, tokenizer=tokenizer, task_name=key, mode="test", cache_dir=data_args.data_cache_dir)
    else:
        extra_gen_datasets = {}

    # Initialize our Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=functools.partial(compute_metrics_fn, tokenizer=tokenizer))

    # --------------------------------------------------------------------------
    # Training.
    # --------------------------------------------------------------------------
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        from transformers import TrainerState
        trainer.state = TrainerState()

    # --------------------------------------------------------------------------
    # Evaluation.
    # --------------------------------------------------------------------------
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results_k={top_k}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        if model_args.eval_all_checkpoints:
            def reinit(cp):
                model = AlbertForTokenRationale.from_pretrained(cp, config=config)
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    eval_dataset=eval_dataset,
                    compute_metrics=functools.partial(compute_metrics_fn, tokenizer=tokenizer))
                return model, trainer

            checkpoints = trainer._sorted_checkpoints()
            best_cp = ""
            highest_mask_f1 = 0
            for cp in checkpoints:
                model, trainer = reinit(cp)
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results {} *****".format(cp))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s: %s = %s\n" % (cp, key, value))
                            if "mask_f1" in key and value > highest_mask_f1:
                                highest_mask_f1 = value
                                best_cp = cp

            with open(output_eval_file, "a") as writer:
                logger.info("***** Best eval mask_f1: {}, {} *****".format(highest_mask_f1, best_cp))
                writer.write("best mask_f1: %s (%s)\n" % (highest_mask_f1, best_cp))

            model, trainer = reinit(best_cp)

    if model_args.do_test:
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        output_eval_file = os.path.join(
            training_args.output_dir, f"test_results_k={top_k}.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        for name, test_dataset in extra_test_datasets.items():
            eval_result = trainer.evaluate(eval_dataset=test_dataset)
            output_eval_file = os.path.join(
                training_args.output_dir, f"test_{name}_results_k={top_k}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Test %s results *****", name)
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            if training_args.do_predict:
                logging.info("*** Predict ***")
                predictions = trainer.predict(test_dataset=test_dataset).predictions
                output_pred_file = os.path.join(
                    training_args.output_dir, f"test_{name}_preds_k={top_k}.pt"
                )
                if trainer.is_world_process_zero():
                    torch.save(predictions, output_pred_file)

    if model_args.gen_on_datasets:
        for name, gen_dataset in extra_gen_datasets.items():
            eval_prediction = trainer.predict(test_dataset=gen_dataset).predictions
            mask = eval_prediction[2]
            is_masked = np.greater(mask, 0.5).astype(np.int)
            input_ids = eval_prediction[6]
            examples = []
            for i in range(len(input_ids)):
                row = tokenizer.convert_ids_to_tokens(input_ids[i])
                original = []
                masked = []
                claim_idx = -1
                for j in range(len(row)):
                    if row[j] in ["[CLS]", "<pad>"]:
                        continue
                    if row[j] == "[SEP]":
                        claim_idx = j + 1
                        break
                    original.append(row[j])
                    if is_masked[i][j]:
                        masked.append("▁<mask>")
                    else:
                        masked.append(row[j])

                claim = []
                if claim_idx >= 0:
                    for j in range(claim_idx, len(row)):
                        if row[j] in ["[CLS]", "<pad>"]:
                            continue
                        if row[j] == "[SEP]":
                            break
                        claim.append(row[j])

                # clean up.
                prev = ""
                cleaned = []
                for j, token in enumerate(masked):
                    if prev == "▁<mask>":
                        if not original[j].startswith("▁"):
                            continue
                    cleaned.append(token)
                    prev = token
                masked = cleaned

                claim = tokenizer.convert_tokens_to_string(claim)
                original = tokenizer.convert_tokens_to_string(original)
                masked = tokenizer.convert_tokens_to_string(masked)
                combined = f"{claim}; EVIDENCE: {original}; PREDS: {masked}"
                examples.append(combined)

            output_gen_file = os.path.join(
                training_args.output_dir, f"test_{name}_gen_k={top_k}.source"
            )
            with open(output_gen_file, "w") as f:
                for example in examples:
                    f.write(example)
                    f.write("\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
