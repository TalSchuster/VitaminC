"""
Process datasets for sentence-pair classification tasks.
Allows combining dataset from multiple data files.
Used for Multi-task fact verification and NLI.
"""

import os
import logging
import time
import random
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers import InputExample, PreTrainedTokenizerBase, InputFeatures

from vitaminc.processing.utils import download_and_extract, convert_examples_to_features
from vitaminc.processing.processor import VitCProcessor


logger = logging.getLogger(__name__)


class VitCFactVerificationProcessor(VitCProcessor):
    def __init__(self, claim_only=False):
        self.claim_only = claim_only

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if 'unique_id' in line:
                guid = line['unique_id']
            else:
                guid = line['id']
            if self.claim_only:
                text_a = line['claim']
                text_b = None
            else:
                text_a = line['evidence']
                text_b = line['claim']

            if 'gold_label' in line:
                label = line['gold_label']
            elif 'label' in line:
                label = line['label']
            else:
                label = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


@dataclass
class VitCDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain subfolders with task names that have jsonlines files."
                          "Cached data files will be saved there (unless --cache_dir is used)."}
    )
    tasks_names: List[str] = field(default=None, metadata={"help": "The names of the tasks to train on."})
    test_tasks: List[str] = field(
        default=None,
        metadata={"help": "The names of the tasks to test on. "
                          "Default is same as train."
                  }
    )
    tasks_ratios: List[float] = field(
        default=None,
        metadata={"help": "Ratios of tasks to use (should sum to 1). "
                          "-1 will use the full dataset for that task (won't be counted towards the dataset_size)."
                            "Use with dataset_size"})
    dataset_size: int = field(
        default=None,
        metadata={
            "help": "Size of dataset to create. Uses tasks_ratios or uniform dist. "
            "Should be <= the size of the full dataset."})
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached data files."}
    )
    claim_only: bool = field(
        default=False, metadata={"help": "Use only the claim sentence from the data"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonlines file containing the training data (overrides data_dir)."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A jsonlines file containing the validation data (overrides data_dir)."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A jsonlines file containing the test data (overrides data_dir)."})

    def __post_init__(self):
        assert not (self.train_file is not None and self.tasks_names is not None), 'When using train_file no need to state task names'
        assert not (self.test_file is not None and self.test_tasks is not None), 'When using test_file no need to state task names'
        if self.tasks_names is not None:
            if self.tasks_ratios is not None:
                assert self.dataset_size is not None, 'tasks_ratios choice should come with a fixed dataset_size'
            else:
                self.tasks_ratios = [1 / len(self.tasks_names)] * len(self.tasks_names)

            assert sum([r for r in self.tasks_ratios if r != -1]) == 1
            assert len(self.tasks_ratios) == len(self.tasks_names)
        else:
            self.tasks_names = []

        if self.test_tasks is None and self.test_file is None:
            # If test_tasks are not declared, use tasks_names
            self.test_tasks = self.tasks_names

        if self.data_dir is None:
            self.data_dir = './data'
        os.makedirs(self.data_dir, exist_ok=True)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class VitCDataset(Dataset):
    args: VitCDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: VitCDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        """
        The dataset will bbe created either from file_path or
        from args.tasks_names if file_path is not given.
        """
        self.args = args
        self.processor = VitCFactVerificationProcessor(claim_only=args.claim_only)
        self.output_mode = "classification"
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        # Load data features from cache or dataset file
        if file_path is None:
            tasks_str = '-'.join(args.tasks_names)
        else:
            tasks_str = hashlib.md5(file_path.encode()).hexdigest()

        if args.claim_only:
            tasks_str += '-claimonly'
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                tasks_str,
                "{}_{}".format('-'.join([str(r) for r in args.tasks_ratios]),
                               args.dataset_size) if args.dataset_size else "all"
            ),
        )
        self.label_list = self.processor.get_labels()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                if mode == Split.dev:
                    get_examples = self.processor.get_dev_examples
                elif mode == Split.test:
                    get_examples = self.processor.get_test_examples
                else:
                    get_examples = self.processor.get_train_examples

                if file_path is not None:
                    examples = self.processor.get_examples_from_file(file_path, mode)
                else:
                    examples = []
                    for task, ratio in zip(args.tasks_names, args.tasks_ratios):
                        task_data_dir = os.path.join(args.data_dir, task)
                        if not os.path.exists(task_data_dir):
                            download_and_extract(task, args.data_dir)
                        logger.info(f"Collecting {mode.value} examples (ratio={ratio}) from dataset file at {task_data_dir}")
                        task_examples = get_examples(task_data_dir)
                        if args.dataset_size is not None:
                            random.shuffle(task_examples)
                            if ratio != -1:
                                task_examples = task_examples[:int(args.dataset_size * ratio)]

                        examples.extend(task_examples)

                self.features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
