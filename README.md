# VitaminC
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
sh scripts/run_fact_verification.sh path_to_test_file
```

Other available pretrained models (including the ALBERT-xlarge model that performed the best):
```
tals/albert-base-vitaminc
tals/albert-base-vitaminc-mnli
tals/albert-base-vitaminc-fever
tals/albert-xlarge-vitaminc
tals/albert-xlarge-vitaminc-mnli
tals/albert-xlarge-vitaminc-fever
```

---
# Word-level Rationales

*Will be added soon*

---

# Factually Consistent Generation

*Will be added soon*

---

# Citation

If you find our code and/or data useful, please cite our paper:

```
@InProceedings{Schuster2019,
    title = "Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence",
    author="Tal Schuster and Adam Fisch and Regina Barzilay",
    booktitle = "NAACL 2021",
    year = "2021",
    url = "https://arxiv.org/abs/2103.08541",
}
```
