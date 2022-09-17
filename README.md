# VitaminC

<p>
<a href="https://console.tiyaro.ai/explore/tals-albert-xlarge-vitaminc-mnli"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>
</p>
This repository contains the dataset and models for the NAACL 2021 paper: [Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence](https://arxiv.org/abs/2103.08541). The VitaminC dataset contains more than 450,000 claim-evidence pairs from over 100,000 revisions to popular Wikipedia pages, and additional "synthetic" revisions.

*We're still updating this repo. More to come soon. Please reach out to us if you have any questions.*

Below are instructions for the four main tasks described in the paper:
* [Revision Flagging](#revision-flagging)
* [Fact Verification](#fact-verification)
* [Word-level Rationales](#word-level-rationales)
* [Factually Consistent Generation](#factually-consistent-generation)

## Install

If you're only interested in the dataset (in jsonlines format), please find the per-task links below.

To install this pacakage with the code to process the dataset and run transformer models and baselines, run:
```
python setup.py install
```
Note: python>=3.7 is needed for all the dependencies to work.

---
# Revision Flagging
VitaminC revision flagging data (the script below will automatically download it): [link](https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc_flagging.zip)

Example of evaluating ALBERT-base model on the test dataset:
```
sh scripts/run_flagging.sh
```

The BOW and edit distance baselines from the paper are in `scripts/factual_flagging_baselines.py`.

---

# Fact Verification

VitaminC fact verification data (the script below will automatically download the required files): [link](https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip)

Example of evaluating ALBERT-base model fine-tuned with VitaminC and FEVER datasets on the "real" and "synthetic" test sets of VitaminC:
```
sh scripts/run_fact_verification.sh
```
To evaluate the same model on another jsonlines file (containing `claim`, `evidence`, and `label` fields). Use:
```
sh scripts/run_fact_verification.sh --test_file path_to_test_file
```

## Finetuned models
Other available pretrained models (including the ALBERT-xlarge model that performed the best):
```
tals/albert-base-vitaminc
tals/albert-base-vitaminc-mnli
tals/albert-base-vitaminc-fever
tals/albert-xlarge-vitaminc
tals/albert-xlarge-vitaminc-mnli
tals/albert-xlarge-vitaminc-fever
```

## Test datasets
The following datasets can be used for testing the models:
```
vitaminc
vitaminc_real
vitaminc_synthetic
fever
mnli
fever_adversarial
fever_symmetric
fever_triggers
anli
```
**Note:** `vitaminc` is a concatanation of `vitaminc_real` and `vitaminc_synthetic`.

**Usage:** provide the desired dataset name with the `--test_tasks` arguemnt as a space-seperated list (for example `--test_tasks vitaminc_real vitaminc_synthetic`).

To compute the test metrics per task, make sure to add `--do_test`. To get the predictions of the model, use the `--do_predict` flag. This will write the predictions and logits to `test_[preds/scores]_{task_name}.txt` files in the `output_dir`.

## Training new models
To train or finetune any transformer from the Hugging Face repository (including farther finetuning the models here), simply add the `--do_train` flag and add the [model name](https://huggingface.co/models) with the `--model_name_or_path` argument.

All of Hugging Face training arguments are available, plus a few added by us:
* `--eval_all_checkpoints`: evaluates the model on all intermediate checkpoints stored during training.
* `--test_on_best_ckpt`: Will run the test/predict using the checkpoint with the best score (instead of the last one).
* `--tasks_names`: a list of training datasets to use for training (see list bellow).
* `--data_dir`: path to dir under which subdirs with names equivalent to `tasks_names` will be stored (to add a new task simply add a subdir with `train/dev/test.jsonl` files that follow the VitaminC data format.
* `--dataset_size`: size of training dataset to use (should be <= size of available data).
* `--tasks_ratios`: a list of task ratios (should sum to 1) to be used when choosing a fixed `dataset_size`. Corresponding to the order of `tasks_names`.
* `--claim_only`: Uses only the claim from the data. 

**Training datasets:** 
The training data from the following datasets will be automatically downloaded when chosen for `tasks_names`:
```
vitaminc
fever
mnli
```

---
# Word-level Rationales

Example of evaluating our distantly supervised ALBERT-base word-level rationale model:
```
sh scripts/run_rationale.sh
```

---

# Factually Consistent Generation

*Will be added soon*

---

# Citation

If you find our code and/or data useful, please cite our paper:

```
@inproceedings{schuster-etal-2021-get,
    title = "Get Your Vitamin {C}! Robust Fact Verification with Contrastive Evidence",
    author = "Schuster, Tal  and
      Fisch, Adam  and
      Barzilay, Regina",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.52",
    pages = "624--643"
}
```
