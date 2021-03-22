import os
from transformers import DataProcessor

from vitaminc.processing.utils import read_jsonlines

class VitCProcessor(DataProcessor):
    def get_examples_from_file(self, file_path, set_type="train"):
        return self._create_examples(
            read_jsonlines(file_path), set_type)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.get_examples_from_file(
            os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.get_examples_from_file(
            os.path.join(data_dir, "dev.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.get_examples_from_file(
            os.path.join(data_dir, "test.jsonl"), "test")

    def get_labels(self):
        """See base class."""
        raise NotImplementedError

    def _create_examples(self, lines, set_type):
        raise NotImplementedError
