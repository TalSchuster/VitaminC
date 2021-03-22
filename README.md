# VitaminC
This repository contains the dataset and model for the NAACL 2021 paper: **Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence**.

*We're still updating this repo. More to come soon. Please reach out to us if you have any questions.*

## Install

If you're only interested in the dataset (in jsonlines format), please find the per-task links below.

To install this pacakage with the code to process the dataset and run transformer models and baselines, run:
```
python setup.py install
```
Note: python>=3.7 is needed for all the dependencies to properly work.

---

# Fact Verification

VitaminC fact verification data: [link](https://www.dropbox.com/s/ivxojzw37ob4nee/vitaminc.zip?dl=0)

Example of evaluating ALBERT-base model fine-tuned on VitaminC on real and synthetic test sets of VitaminC:
```
sh scripts/run_fact_verification.sh
```
To evaluate the same model on another jsonlines file (containing `claim`, `evidence`, and `label` fields). Use:
```
sh scripts/run_fact_verification.sh path_to_test_file
```

---

# Citation

If you find our code and/or data useful, please cite our paper.
