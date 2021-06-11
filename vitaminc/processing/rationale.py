"""Process datasets for token tagging rationale."""

import jsonlines
import logging
import os
import time
import tqdm

from dataclasses import dataclass
from typing import List, Optional, Union
from filelock import FileLock

import torch
from torch.utils.data.dataset import Dataset
from transformers import DataProcessor

from vitaminc.processing.utils import download_and_extract

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    mask_labels: Optional[List[int]] = None


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    evidence_mask: Optional[List[float]] = None
    label: Optional[Union[int, float]] = None
    mask_labels: Optional[List[int]] = None
    is_unsupervised: Optional[float] = 1.0


class RationaleProcessor(DataProcessor):

    @staticmethod
    def _read_jsonlines(input_file):
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            reader = jsonlines.Reader(f)
            for line in reader.iter(type=dict):
                lines.append(line)
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonlines(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if "unique_id" in line:
                guid = line["unique_id"]
            else:
                guid = line["id"]

            text_b = line["claim"]
            text_a = line["evidence"]

            if not text_a or not text_b:
                continue

            if "orig_label" in line:
                label = line["orig_label"]
            elif "gold_label" in line:
                label = line["gold_label"]
            elif "label" in line:
                label = line["label"]
            else:
                label = None

            if "masked_inds" in line:
                mask_labels = line["masked_inds"]
            else:
                mask_labels = []

            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    mask_labels=mask_labels))

        return examples


class RationaleDataset(Dataset):

    def __init__(self, args, tokenizer, task_name=None, mode="train", cache_dir=None):
        self.args = args
        self.processor = RationaleProcessor()
        task_name = task_name if task_name else os.path.basename(args.data_dir.rstrip("/"))

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_rationale_{}_{}_{}".format(
                task_name,
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
            ),
        )
        self.label_list = self.processor.get_labels()

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                task = "vitaminc_rationale"
                task_data_dir = os.path.join(args.data_dir, task)
                if not os.path.exists(task_data_dir):
                    download_and_extract(task, args.data_dir)
                if mode == "dev":
                    examples = self.processor.get_dev_examples(task_data_dir)
                elif mode == "test":
                    examples = self.processor.get_test_examples(task_data_dir)
                elif mode == "train":
                    examples = self.processor.get_train_examples(task_data_dir)
                else:
                    raise ValueError("mode is not a valid split name")

                self.features = convert_fn(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=self.label_list,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]",
                    cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.label_list


def convert_fn(examples, tokenizer, label_list, max_length=None, verbose=True):
    """Returns a list of processed features."""
    if max_length is None:
        max_length = tokenizer.max_len

    logger.info("Using label list %s" % (label_list))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for example in tqdm.tqdm(examples, "converting to features"):
        mask_labels = []
        all_evidence_tokens = []
        for (i, token) in enumerate(example.text_a.split()):
            for subtoken in tokenizer.tokenize(token):
                if example.mask_labels:
                    if i in example.mask_labels:
                        mask_labels.append(1)
                    else:
                        mask_labels.append(0)
                all_evidence_tokens.append(subtoken)
        all_claim_tokens = [sub_t for t in example.text_b.split() for sub_t in tokenizer.tokenize(t)]

        encoding = tokenizer.encode_plus(
            text=all_evidence_tokens,
            text_pair=all_claim_tokens,
            max_length=max_length,
            padding="max_length",
            truncation=True)

        num_special = tokenizer.num_special_tokens_to_add(pair=True)
        evidence_mask = [0.0] + [1.0] * min(max_length - num_special, len(all_evidence_tokens))
        evidence_mask = evidence_mask + [0.0] * (max_length - len(evidence_mask))

        if mask_labels:
            is_unsupervised = 0
            mask_labels = mask_labels[:max_length - num_special]
            mask_labels = [-1] + mask_labels + [-1] * (max_length - len(mask_labels) - 1)
        else:
            is_unsupervised = 1
            mask_labels = [-1] * max_length

        assert(len(evidence_mask) == max_length)
        assert(len(mask_labels) == max_length)

        if example.label:
            label = label_map[example.label]
        else:
            label = None

        feature = InputFeatures(
            **encoding,
            evidence_mask=evidence_mask,
            label=label,
            mask_labels=mask_labels,
            is_unsupervised=is_unsupervised)
        features.append(feature)

    if verbose:
        for i, example in enumerate(examples[:5]):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features[i])

    return features
