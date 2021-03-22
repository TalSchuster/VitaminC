"""
Process datasets for flagging factual revisions.
"""

import os
import logging
import time
import random
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


class FlaggingProcessor(VitCProcessor):
    '''
    Prepare data for factual/ not factual classification based on the
    sentences before and after the revision.
    '''
    def get_labels(self):
        """See base class."""
        return ["not factual", "factual"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line['case_id']
            text_a = line['sent_a']
            text_b = line['sent_b']
            if 'label' in line:
                label = line['label']
            else:
                label = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class FlaggingProcessorDiffViz(FlaggingProcessor):
    '''
    Use the diff visulaization (single sentence).
    '''
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line['case_id']
            text_a = line['diff_viz']
            if 'label' in line:
                label = line['label']
            else:
                label = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FlaggingProcessorDiffTokens(FlaggingProcessor):
    '''
    Use only the tokens that are different between the two sentences.
    '''
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line['case_id']
            text_a = line['sent_a_diff']
            text_b = line['sent_b_diff']
            if 'label' in line:
                label = line['label']
            else:
                label = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


processors = {
    "raw_sents": FlaggingProcessor,
    "diff_viz": FlaggingProcessorDiffViz,
    "diff_tokens": FlaggingProcessorDiffTokens,
}


@dataclass
class FlaggingDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .jsonl files."}
    )
    input_features: str = field(
        default="raw_sents",
        metadata={"help": "What input to provide for the classifier. options: %s." % list(processors.keys())}
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        assert self.input_features in processors.keys()
        if self.data_dir is None:
            self.data_dir = './data'
        os.makedirs(self.data_dir, exist_ok=True)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class FlaggingDataset(Dataset):
    args: FlaggingDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: FlaggingDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
        file_path: Optional[str] = None,
    ):
        self.args = args
        self.processor = processors[args.input_features]()
        self.output_mode = "classification"
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        task = "vitaminc_flagging"
        # Load data features from cache or dataset file
        if file_path is None:
            tasks_str = task
        else:
            tasks_str = hashlib.md5(file_path.encode()).hexdigest()

        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                tasks_str,
                args.input_features,
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
                    logger.info(f"Collecting {mode.value} examples from {file_path}")
                    examples = self.processor.get_examples_from_file(file_path, mode)
                else:
                    task_data_dir = os.path.join(args.data_dir, task)
                    if not os.path.exists(task_data_dir):
                        download_and_extract(task, args.data_dir)
                    logger.info(f"Collecting {mode.value} examples from {task_data_dir}")
                    examples = get_examples(task_data_dir)

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
