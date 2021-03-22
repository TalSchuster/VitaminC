import os
import urllib.request as request
import zipfile
import logging
import jsonlines
from typing import List, Optional, Union

from transformers import InputExample, PreTrainedTokenizer, InputFeatures

logger = logging.getLogger(__name__)


TASK2PATH = {
    "vitaminc" : "https://www.dropbox.com/s/ivxojzw37ob4nee/vitaminc.zip?dl=1",
    "vitaminc_real" : "https://www.dropbox.com/s/y74sez0qa8z5dnw/vitaminc_real.zip?dl=1",
    "vitaminc_synthetic" : "https://www.dropbox.com/s/uqb2316chhtx77z/vitaminc_synthetic.zip?dl=1",
    "vitaminc_flagging": "https://www.dropbox.com/s/xcwiy8vg04xvbtb/vitaminc_flagging.zip?dl=1",
}


def download_and_extract(task, data_dir):
    if task not in TASK2PATH:
        logger.warning("No stored url for task %s. Please download manually." % task)
        return
    logger.info("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    logger.info("Completed! Stored at %s" % data_dir)


def read_jsonlines(input_file):
    lines = []
    with open(input_file, "r", encoding='utf-8') as f:
        reader = jsonlines.Reader(f)
        for line in reader.iter(type=dict):
            lines.append(line)

    return lines


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b)
         if example.text_b is not None else example.text_a
         for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features
