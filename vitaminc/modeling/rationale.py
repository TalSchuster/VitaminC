"""Custom rationale model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from sklearn import metrics

from transformers.file_utils import ModelOutput
from transformers import (
    AlbertForSequenceClassification,
    AlbertModel,
    AlbertPreTrainedModel,
    )


def compute_metrics_fn(eval_prediction, tokenizer=None, return_text=False, nei_index=2):
    full_logits = eval_prediction.predictions[0]
    masked_logits = eval_prediction.predictions[1]
    mask = eval_prediction.predictions[2]
    labels = eval_prediction.predictions[3]
    mask_labels = eval_prediction.predictions[4]
    mask_labels = np.maximum(mask_labels, 0)

    full_preds = np.argmax(full_logits, axis=1)
    masked_preds = np.argmax(masked_logits, axis=1)
    is_masked = np.greater(mask, 0.5).astype(np.int)

    results = {}
    results["full_acc"] = metrics.accuracy_score(labels, full_preds)
    results["masked_acc"] = metrics.accuracy_score(np.ones_like(labels) * nei_index, masked_preds)
    results["mask_f1"] = metrics.f1_score(mask_labels, is_masked, average="micro", zero_division=1)
    results["mask_recall"] = metrics.recall_score(mask_labels, is_masked, average="micro", zero_division=1)
    results["mask_precision"] = metrics.precision_score(mask_labels, is_masked, average="micro", zero_division=1)

    if return_text:
        input_ids = eval_prediction.predictions[6]
        examples = []
        for i in range(len(input_ids)):
            import pdb; pdb.set_trace()
            row = tokenizer.convert_ids_to_tokens(input_ids[i])
            original = []
            masked = []
            for j in range(len(row)):
                if row[j] in ["[CLS]", "[SEP]", "<pad>"]:
                    continue
                original.append(row[j])
                if is_masked[i][j]:
                    masked.append("▁<mask>")
                else:
                    masked.append(row[j])
            original = tokenizer.convert_tokens_to_string(original)
            masked = tokenizer.convert_tokens_to_string(masked)
            combined = f"{original} OLD: {masked}"
            examples.append(combined)
        results["text"] = examples

    return results


class MaskOutput(ModelOutput):
    mask: torch.FloatTensor
    loss: torch.FloatTensor


class ClassifyOutput(ModelOutput):
    logits: torch.FloatTensor
    loss: torch.FloatTensor


class TokenTaggingRationaleOutput(ModelOutput):
    loss: torch.FloatTensor
    full_logits: torch.FloatTensor
    masked_logits: torch.FloatTensor
    mask: torch.FloatTensor
    labels: Optional[torch.FloatTensor] = None
    mask_labels: Optional[torch.FloatTensor] = None
    is_unsupervised: Optional[torch.FloatTensor] = None
    input_ids: Optional[torch.LongTensor] = None


class AlbertForTokenRationale(AlbertPreTrainedModel):

    def __init__(self, config):
        super(AlbertForTokenRationale, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.nei_index = config.nei_index
        self.mask_token_id = config.mask_token_id
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.masker = nn.Linear(config.hidden_size, 2)
        if not self.config.use_pretrained_classifier:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.init_weights()
        else:
            self.init_weights()
            self.classifier = AlbertForSequenceClassification.from_pretrained(config.pretrained_classifier_path)
            if self.config.fix_pretrained_classifier:
                for p in self.classifier.parameters():
                    p.requires_grad = False

    def mask(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        evidence_mask=None,
        labels=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        sequence_output = outputs[0]
        logits = self.masker(self.dropout(sequence_output))

        # Mask non-evidence tokens.
        if evidence_mask is not None:
            logits[:, :, 1] = logits[:, :, 1] - (1 - evidence_mask) * 1e10

        # Get supervised loss for polarizing tokens.
        loss = 0
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            loss = loss * labels.view(-1).ne(-1).float()
            loss = loss.view(logits.size(0), logits.size(1))
            loss = loss.sum(dim=1) / (labels.ne(-1).sum(dim=1) + 1e-12)

        if self.training:
            # Take gumbel softmax of mask logits.
            mask_weights = F.gumbel_softmax(logits, tau=self.config.temperature).select(2, 1)
        else:
            # Take hard max of mask logits. Limit to k if specified.
            if not self.config.top_k >= 0:
                mask_weights = torch.argmax(logits, dim=-1).float()
            else:
                # Get kth value.
                kth_value = -torch.kthvalue(-logits, k=self.config.top_k, dim=1).values[:, 1]
                kth_value = kth_value.view(-1, 1)

                # Nix everything below kth value. Boost everything above kth value.
                mask_logits = logits[:, :, 1]
                mask_logits += (mask_logits.ge(kth_value).float() -
                                mask_logits.lt(kth_value).float()) * 1e8
                mask_weights = torch.argmax(logits, dim=-1).float()

        return MaskOutput(
            mask=mask_weights,
            loss=loss
        )

    def classify(
        self,
        input_ids=None,
        mask_weights=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        def embed(ids):
            if not self.config.use_pretrained_classifier:
                embeds = self.albert.get_input_embeddings()(ids)
            else:
                embeds = self.classifier.albert.get_input_embeddings()(ids)
            return embeds

        # Embed inputs.
        input_embeds = embed(input_ids)

        # Targeted mask.
        mask_ids = input_ids.clone().fill_(self.mask_token_id)
        mask_embeds = embed(mask_ids)
        mask_weights = mask_weights.unsqueeze(-1)

        # Mix embeddings.
        mix_embeds = mask_weights * mask_embeds + (1 - mask_weights) * input_embeds

        # Run model.
        if not self.config.use_pretrained_classifier:
            outputs = self.albert(
                inputs_embeds=mix_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids)
            pooled_output = outputs[1]
            logits = self.classifier(self.dropout(pooled_output))
        else:
            outputs = self.classifier(
                inputs_embeds=mix_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                return_dict=True)
            logits = outputs.logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ClassifyOutput(
            logits=logits,
            loss=loss,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        evidence_mask=None,
        labels=None,
        mask_labels=None,
        is_unsupervised=None,
        **kwargs,
    ):
        # Get example weights.
        if is_unsupervised is None:
            is_unsupervised = torch.zeros(input_ids.size(0)).to(input_ids.device)
        example_weights = 1.0 * (1 - is_unsupervised) + self.config.unsupervised_weight * is_unsupervised

        # Encode the inputs and predict token-level masking scores.
        mask_output = self.mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            evidence_mask=evidence_mask,
            labels=mask_labels)
        mask_supervised_loss = mask_output.loss if mask_labels is not None else 0
        mask_weights = mask_output.mask

        # Randomly mask if training.
        if self.training:
            rand_mask_weights = torch.rand_like(mask_weights)
            rand_mask_weights = rand_mask_weights.lt(0.1).float()
            # permute = np.random.permutation(mask_weights.size(1))
            # permute = torch.from_numpy(permute).to(mask_weights.device)
            # rand_mask_weights = mask_weights.index_select(dim=1, index=permute)
            rand_mask_weights *= evidence_mask
        else:
            rand_mask_weights = torch.zeros_like(mask_weights)

        # Get the output with random masking.
        full_cls_output = self.classify(
            input_ids=input_ids,
            mask_weights=rand_mask_weights.detach(),
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels)
        full_cls_loss = full_cls_output.loss if labels is not None else 0

        # Get the output with targeted masking.
        masked_cls_output = self.classify(
            input_ids=input_ids,
            mask_weights=mask_weights,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels.clone().fill_(self.nei_index) if labels is not None else None)
        masked_cls_loss = masked_cls_output.loss if labels is not None else 0

        # Compute (soft) number of masked tokens
        mask_weights = mask_output.mask
        total = attention_mask.sum(dim=1) + 1e-12
        num_masked = mask_weights.sum(dim=1)
        sparsity_loss = num_masked / total

        # Compute (soft) number of discontinuities.
        num_transitions = torch.abs(mask_weights[:, 1:] - mask_weights[:, :-1])
        num_transitions = torch.sum(num_transitions * attention_mask[:, 1:], dim=1)
        continuity_loss = num_transitions / total

        # Add loss components.
        loss = (self.config.confusion_weight * masked_cls_loss +
                self.config.supervised_weight * mask_supervised_loss +
                self.config.continuity_weight * continuity_loss +
                self.config.sparsity_weight * sparsity_loss)
        loss = labels.ne(self.nei_index) * loss + full_cls_loss
        loss = example_weights * loss
        loss = loss.mean()

        return TokenTaggingRationaleOutput(
            loss=loss,
            full_logits=full_cls_output.logits,
            masked_logits=masked_cls_output.logits,
            mask=mask_output.mask,
            labels=labels,
            mask_labels=mask_labels,
            is_unsupervised=is_unsupervised,
            input_ids=input_ids)
