import os
import argparse
import jsonlines
import torch
import collections
import subprocess
import nltk
import shutil
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

Example = collections.namedtuple(
    "Example", ["bow_a", "bow_b", "label"])

UNK= 'UNK'


def construct_batch(batch):
    emb_a = torch.stack([x.bow_a for x in batch])
    emb_b = torch.stack([x.bow_b for x in batch])
    labels = torch.Tensor([x.label for x in batch]).long()

    inputs = [emb_a, emb_b, labels]
    return inputs


class FlaggingBOWClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # Load embeddings
        self.vocab = dict()
        if not os.path.exists(hparams.embedding_path):
            os.makedirs(os.path.dirname(hparams.embedding_path), exist_ok=True)
            subprocess.check_output(["wget", "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec", "-O", hparams.embedding_path])
        num_lines = int(subprocess.check_output(["wc", "-l", hparams.embedding_path], encoding="utf8").split()[0])
        with open(hparams.embedding_path, 'r') as f:
            for i, line in tqdm(enumerate(f), total=num_lines, desc="Load embeddings"):
                if i == 0:
                   self.vocab_size, self.emb_dim = int(line.split()[0]), int(line.split()[1])
                   self.embedding = torch.zeros(self.vocab_size + 1, self.emb_dim)
                   continue
                word = line.split()[0]
                emb = torch.tensor([float(x) for x in line.strip().split(' ')[1:]])
                if len(emb) != self.emb_dim:
                    continue
                self.embedding[i-1,:] = emb
                self.vocab[word] = i - 1

        # The OOV embedding is the average of all embeddings.
        self.embedding[self.vocab_size, :] = self.embedding[:-1,:].mean(0)
        self.vocab[UNK] = self.vocab_size

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim * 3 + 1, hparams.cls_hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.cls_hidden_size, hparams.cls_hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.cls_hidden_size, 2)
        )

    def forward(self, x_a, x_b):
        abs_diff = torch.abs(x_a - x_b)
        dot_prod = torch.mm(x_a,x_b.T).diag().unsqueeze(1)
        x = torch.cat((x_a, x_b, abs_diff, dot_prod), dim=1)
        logits = self.classifier(x)
        return logits

    def comp_loss_and_acc(self, logits, y):
        loss = F.cross_entropy(logits, y)
        acc = sum(logits.argmax(1) == y).true_divide(logits.size(0))
        return loss, acc

    def training_step(self, batch, batch_idx):
        x_a, x_b, y = batch
        logits = self.forward(x_a, x_b)

        loss, acc = self.comp_loss_and_acc(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x_a, x_b, y = batch
        logits = self.forward(x_a, x_b)

        loss, acc = self.comp_loss_and_acc(logits, y)
        return {"loss": loss, "acc": acc, "logits": logits, "labels": y}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["loss"].unsqueeze(0).detach() for x in outputs], dim=0).mean()
        acc = torch.cat([x["acc"].unsqueeze(0).detach() for x in outputs], dim=0).mean()
        logits = torch.cat([x["logits"].detach() for x in outputs], dim=0)
        labels = torch.cat([x["labels"].detach() for x in outputs], dim=0)
        probs = torch.softmax(logits, dim=1)
        roc_auc = metrics.roc_auc_score(labels.cpu().numpy(), probs[:,1].cpu().numpy())
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_roc_auc', roc_auc)

    def test_epoch_end(self, outputs):
        loss = torch.cat([x["loss"].unsqueeze(0).detach() for x in outputs], dim=0).mean()
        acc = torch.cat([x["acc"].unsqueeze(0).detach() for x in outputs], dim=0).mean()
        logits = torch.cat([x["logits"].detach() for x in outputs], dim=0)
        labels = torch.cat([x["labels"].detach() for x in outputs], dim=0)
        probs = torch.softmax(logits, dim=1)
        roc_auc = metrics.roc_auc_score(labels.cpu().numpy(), probs[:,1].cpu().numpy())
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_roc_auc', roc_auc)

        self.test_probs = probs[:,1].cpu().numpy()
        self.test_labels = labels.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=2, min_lr=1e-6),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def load_dataset(self, dataset_file):
        '''
        Loads the jsonlines file and copmutes the average of the embeddings
        of the words that are different in sent_a and sent_b.
        '''
        def encode_diff(diff_words):
            words = diff_words.replace(' ; ', ' ')
            words = nltk.word_tokenize(words)
            words = [w if w in self.vocab else UNK for w in words]
            inds = [self.vocab[w] for w in words]
            return inds

        def avg_embeds(inds):
            if len(inds) > 0:
                embeds = self.embedding[torch.LongTensor(inds), :]

                # Average embeddings.
                bow = embeds.mean(0)
            else:
                bow = torch.zeros(self.emb_dim)

            return bow

        dataset = []
        print(f"Loading {dataset_file}")
        num_lines = int(subprocess.check_output(["wc", "-l", dataset_file], encoding="utf8").split()[0])
        with open(dataset_file, "r", encoding='utf-8') as f:
            reader = jsonlines.Reader(f)
            for line in tqdm(reader.iter(type=dict), total=num_lines):
                sent_a_words = line['sent_a_diff']
                inds_a = encode_diff(sent_a_words)
                bow_a = avg_embeds(inds_a)

                sent_b_words = line['sent_b_diff']
                inds_b = encode_diff(sent_b_words)
                bow_b = avg_embeds(inds_b)

                label = 1 if line['label'] == "factual" else 0

                dataset.append(Example(bow_a, bow_b, label))

        return dataset

    def prepare_data(self):
        for split in ["train", "val", "test"]:
            dataset = self.load_dataset(
                dataset_file=getattr(self.hparams, "%s_data" % split))
            setattr(self, "%s_dataset" % split, dataset)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_batch)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_batch)
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_batch)
        return loader


    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.register("type", "bool", pl.utilities.parsing.str_to_bool)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--gpus", type=int, nargs="+", default=None)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--checkpoint_dir", type=str, default="results/vitaminc_flagging_bow")
        parser.add_argument("--overwrite", type="bool", default=True)

        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--max_epochs", type=int, default=10)
        parser.add_argument("--num_data_workers", type=int, default=20)
        parser.add_argument("--embedding_path", type=str, default="fasttext/wiki.multi.en.vec")

        parser.add_argument("--train_data", type=str, default="data/vitaminc_flagging/train.jsonl")
        parser.add_argument("--val_data", type=str, default="data/vitaminc_flagging/dev.jsonl")
        parser.add_argument("--test_data", type=str, default="data/vitaminc_flagging/test.jsonl")

        parser.add_argument("--cls_hidden_size", type=int, default=64)

        parser.add_argument("--run_distance_baseline", type="bool", default=False)

        return parser


def compute_metrics_fn(probs, label_ids):
    preds = probs > 0.5
    acc = (preds == label_ids).mean()
    prec, rec, f1, _ = metrics.precision_recall_fscore_support(label_ids, probs>0.5, average="macro")
    roc_auc = metrics.roc_auc_score(label_ids, probs)
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": f1,
        "AUC": roc_auc,
    }


def distance_baseline(dev_file, test_file, output_dir):
    # Train LR on dev distances and predict on test.
    from sklearn.linear_model import LogisticRegression
    try:
        from Levenshtein import distance
        from nltk.corpus import stopwords
    except:
        print("Install nltk and Levenshtein packages to run distance baselines")
        return

    print("Computing distance baseline...")

    gold_labels = []
    dists = []
    with open(test_file, 'r') as f:
        reader = jsonlines.Reader(f)
        for i, line in tqdm(enumerate(reader)):
            gold_labels.append(1 if line['label'] == 'factual' else 0)
            ad = ' '.join([w for w in line['sent_a_diff'].split() if w not in stopwords.words('english')])
            bd = ' '.join([w for w in line['sent_b_diff'].split() if w not in stopwords.words('english')])
            dists.append(distance(ad, bd))

    gold_labels = np.array(gold_labels)
    dists = np.array(dists)

    dev_gold_labels = []
    dev_dists = []
    with open(dev_file, 'r') as f:
        reader = jsonlines.Reader(f)
        for i, line in tqdm(enumerate(reader)):
            dev_gold_labels.append(1 if line['label'] == 'factual' else 0)
            ad = ' '.join([w for w in line['sent_a_diff'].split() if w not in stopwords.words('english')])
            bd = ' '.join([w for w in line['sent_b_diff'].split() if w not in stopwords.words('english')])
            dev_dists.append(distance(ad, bd))

    dev_gold_labels = np.array(dev_gold_labels)
    dev_dists = np.array(dev_dists)

    lr = LogisticRegression(penalty='l2', class_weight='balanced')

    lr.fit(dev_dists.reshape(-1, 1), dev_gold_labels)

    y = lr.predict(dists.reshape(-1, 1))
    s = lr.predict_proba(dists.reshape(-1, 1))[:,1]

    prec, rec, f1, _ = metrics.precision_recall_fscore_support(gold_labels, y, average="macro")
    print(f"name=dist:\n {metrics.roc_auc_score(gold_labels, s)*100:.2f} & {prec*100:.2f} & {rec*100:.2f} & {f1*100:.2f}")
    results = compute_metrics_fn(s, gold_labels)
    with open(os.path.join(args.checkpoint_dir, "results_edit_distance.txt"), "w") as writer:
        print("***** Test results - Edit Distance *****")
        for key, value in results.items():
            print("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))


def main(args):
    pl.seed_everything(args.seed)
    #model = FlaggingBOWClassifier(hparams=args)
    #print(model)
    #if os.path.exists(args.checkpoint_dir) and os.listdir(args.checkpoint_dir):
    #    if not args.overwrite:
    #        raise RuntimeError("Experiment directory is not empty.")
    #    else:
    #        shutil.rmtree(args.checkpoint_dir)
    #checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #    filepath=os.path.join(args.checkpoint_dir, "weights.pt"),
    #    verbose=True,
    #    save_top_k=1,
    #    monitor="val_loss",
    #    mode="min")
    #logger = pl.loggers.tensorboard.TensorBoardLogger(
    #    save_dir=os.path.join(args.checkpoint_dir, "logs"))
    #trainer = pl.Trainer(
    #    logger=logger,
    #    checkpoint_callback=checkpoint_callback,
    #    gpus=args.gpus,
    #    max_epochs=args.max_epochs,
    #    terminate_on_nan=True)
    #trainer.fit(model)
    #results = trainer.test()
    #np.save(os.path.join(args.checkpoint_dir, "test_labels.npy"), model.test_labels)
    #np.save(os.path.join(args.checkpoint_dir, "test_probs.npy"), model.test_probs)
    #results = compute_metrics_fn(model.test_probs, model.test_labels)
    #with open(os.path.join(args.checkpoint_dir, "results.txt"), "w") as writer:
    #    print("***** Test results *****")
    #    for key, value in results.items():
    #        print("  %s = %s", key, value)
    #        writer.write("%s = %s\n" % (key, value))

    if args.run_distance_baseline:
        distance_baseline(args.val_data, args.test_data, args.checkpoint_dir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser = FlaggingBOWClassifier.add_argparse_args(argparser)
    args = argparser.parse_args()
    main(args)
