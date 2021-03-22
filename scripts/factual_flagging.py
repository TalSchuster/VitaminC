import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
from sklearn import metrics
from copy import deepcopy

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from vitaminc.processing.flagging import (
    FlaggingProcessor,
    FlaggingDataTrainingArguments,
    FlaggingDataset,
    )


logger = logging.getLogger(__name__)


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


@dataclass
class TrainingArgs(TrainingArguments):
    eval_all_checkpoints: bool = field(
        default=False, metadata={"help": "Run evaluation on all checkpoints."}
    )
    do_test: bool = field(
        default=False, metadata={"help": "Run evaluation on test set (needs labels)."}
    )
    test_on_best_ckpt: bool = field(
        default=False, metadata={"help": "Load best ckpt for testing (after running with eval_all_checkpoints)."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, FlaggingDataTrainingArguments, TrainingArgs))

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
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

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

    num_labels = len(FlaggingProcessor().get_labels())

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="flagging",
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        FlaggingDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        FlaggingDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )

    def compute_metrics_fn(p: EvalPrediction):
        softmax = lambda x: np.exp(x)/sum(np.exp(x))
        probs = np.array([softmax(s)[1] for s in p.predictions])
        preds = np.argmax(p.predictions, axis=1)
        acc = (preds == p.label_ids).mean()
        prec, rec, f1, _ = metrics.precision_recall_fscore_support(p.label_ids, probs>0.5, average="macro")
        roc_auc = metrics.roc_auc_score(p.label_ids, probs)
        #f1 = metrics.f1_score(y_true=p.label_ids, y_pred=preds, average="macro", labels=np.unique(p.label_ids))
        return {
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "macro_f1": f1,
            "AUC": roc_auc,
        }

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    output_best_ckpt_file = os.path.join(
        training_args.output_dir, "eval_best_ckpt.txt"
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results.txt"
        )
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)

        if training_args.eval_all_checkpoints:
            checkpoints = trainer._sorted_checkpoints()
            best_ckpt = ''
            highest_acc = 0
            for ckpt in checkpoints:
                model = AutoModelForSequenceClassification.from_pretrained(ckpt)
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics_fn
                )
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results : {} *****".format(ckpt))
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s: %s = %s\n" % (ckpt, key, value))
                            if 'acc' in key and value > highest_acc:
                                highest_acc = value
                                best_ckpt = ckpt

            with open(output_eval_file, "a") as writer:
                logger.info("***** Best eval accuracy: {}, '{}' *****".format(highest_acc, best_ckpt))
                writer.write("best acc: %s ('%s')\n" % (highest_acc, best_ckpt))
            with open(output_best_ckpt_file, "w") as writer:
                writer.write("%s" % best_ckpt)

            model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_fn
            )

    if training_args.do_test or training_args.do_predict:
        if training_args.test_on_best_ckpt:
            with open(output_best_ckpt_file, "r") as reader:
                best_ckpt = reader.readlines()[0].strip()
                logger.info("Loading best model from %s", best_ckpt)
                model = AutoModelForSequenceClassification.from_pretrained(best_ckpt)

        test_dataset = FlaggingDataset(
                            data_args,
                            tokenizer=tokenizer,
                            mode="test",
                            cache_dir=model_args.cache_dir
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics_fn
        )
        if training_args.do_predict:
            logger.info("Predicting...")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            pred_labels = np.argmax(predictions, axis=1)
        if training_args.do_test:
            # TODO: Use predictions form do_predict for evaluation.
            logger.info("Evaluating...", )
            eval_result = trainer.evaluate(eval_dataset=test_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, f"test_results.txt"
        )
        output_pred_file = os.path.join(
            training_args.output_dir, f"test_preds.txt"
        )
        output_scores_file = os.path.join(
            training_args.output_dir, f"test_scores.txt"
        )
        if trainer.is_world_process_zero():
            if training_args.do_test:
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Test results *****")
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            if training_args.do_predict:
                with open(output_pred_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(pred_labels):
                        item = test_dataset.get_labels()[item]
                        writer.write("%d\t%s\n" % (index, item))

                with open(output_scores_file, "w") as writer:
                    writer.write("index\t%s\n" % "\t".join(test_dataset.get_labels()))
                    for index, scores in enumerate(predictions):
                        writer.write("%d\t%s\n" % (index, "\t".join([str(x) for x in scores])))

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
