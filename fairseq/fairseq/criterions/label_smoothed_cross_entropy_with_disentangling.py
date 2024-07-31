import math
import numpy as np

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils

from collections import deque

import torch
import torch.nn as nn
# import pdb


@register_criterion("label_smoothed_cross_entropy_with_disentangling")
class LabelSmoothedCrossEntropyCriterionWithDisentangling(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False,
                 temperature=0.1,
                 disentangler_lambda=0.1,
                 disentangler_negative_lambda=0.2,
                 disentangler_reconstruction_lambda=0.1):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.temperature = temperature
        self.disentangler_lambda = disentangler_lambda
        self.disentangler_negative_lambda = disentangler_negative_lambda
        self.disentangler_reconstruction_lambda = disentangler_reconstruction_lambda
        
    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--temperature", type=float,
                            default=0.1)
        parser.add_argument("--disentangler-lambda", type=float,
                            default=0.1)
        parser.add_argument("--disentangler-negative-lambda", type=float,
                            default=0.2)
        parser.add_argument("--disentangler-reconstruction-lambda", type=float,
                            default=0.1)
    
    def swap_sample(self, sample):
        target = sample["target"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        src_tokens = torch.cat((prev_output_tokens[:, :1], sample["net_input"]['src_tokens']), dim=-1)
        
        return {
            "net_input": {
                "src_tokens": target.contiguous(),
                "src_lengths": (target != self.padding_idx).int().sum(dim=1),
                "prev_output_tokens": src_tokens[:, :-1].contiguous()
            },
            'nsentences': sample['nsentences'],
            'ntokens': utils.item((src_tokens[:, 1:] != self.padding_idx).int().sum().data),
            "target": src_tokens[:, 1:].contiguous(),
            "id": sample["id"],
        }
    
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        encoder_out = model.encoder.forward(sample["net_input"]["src_tokens"], sample["net_input"]["src_lengths"])
        encoder_out_sem, encoder_out_lang = model.disentangler.forward(encoder_out)
        
        reverse_sample = self.swap_sample(sample)
        reverse_encoder_out = model.encoder.forward(reverse_sample["net_input"]["src_tokens"], reverse_sample["net_input"]["src_lengths"])
        reverse_encoder_out_sem, reverse_encoder_out_lang = model.disentangler.forward(reverse_encoder_out)
                
        det_loss = self.get_det_loss(
            sample,
            reverse_sample,
            encoder_out["encoder_out"][0],
            reverse_encoder_out["encoder_out"][0],
            encoder_out_sem,
            reverse_encoder_out_sem,
            encoder_out_lang,
            reverse_encoder_out_lang
        )
        
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        all_loss = loss + det_loss * self.disentangler_lambda * ntokens
        
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if isinstance(det_loss, int):
            logging_output["det_loss"] = 0
        else:
            logging_output["det_loss"] = utils.item(det_loss.data)

        return all_loss, sample_size, logging_output
    
    def similarity_function(self, ):
        return nn.CosineSimilarity(dim=-1)
    
    def get_det_loss(self, sample, reverse_sample, encoder_out, reverse_encoder_out, encoder_out_sem, reverse_encoder_out_sem, encoder_out_lang, reverse_encoder_out_lang):
        
        # output the average of hidden states
        def _sentence_embedding(encoder_out, sample):
            encoder_output = encoder_out.transpose(0, 1)
            src_tokens = sample["net_input"]["src_tokens"]
            mask = (src_tokens != self.padding_idx)

            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # (batch, embed_dim)
            return encoder_embedding
        
        def _reconstruct_loss(encoder_embedding, reverse_encoder_embedding, encoder_embedding_sem, reverse_encoder_embedding_sem, encoder_embedding_lang, reverse_encoder_embedding_lang):
            batch_size = encoder_embedding.shape[0]
            embed_dim = encoder_embedding.shape[1]
            
            reconstruct_encoder_out_loss = nn.PairwiseDistance(p=2)(encoder_embedding, encoder_embedding_sem + encoder_embedding_lang) / embed_dim
            reconstruct_encoder_out_loss = reconstruct_encoder_out_loss.sum() / batch_size
            reconstruct_reverse_encoder_out_loss = nn.PairwiseDistance(p=2)(reverse_encoder_embedding, reverse_encoder_embedding_sem + reverse_encoder_embedding_lang) / embed_dim
            reconstruct_reverse_encoder_out_loss = reconstruct_reverse_encoder_out_loss.sum() / batch_size
                        
            return reconstruct_encoder_out_loss + reconstruct_reverse_encoder_out_loss
        
        def _semantic_cosin_loss(encoder_embedding_sem, reverse_encoder_embedding_sem):
            batch_size = encoder_embedding_sem.shape[0]
            embed_dim = encoder_embedding_sem.shape[1]
            anchor_feature = encoder_embedding_sem         # source
            contrast_feature = reverse_encoder_embedding_sem       # target
            
            similarity_function = self.similarity_function()
            # anchor_dot_contrast: cos similarity of source sentences to target sentences, shape: (batch, batch)
            anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, embed_dim)),
                                                    torch.transpose(contrast_feature.expand((batch_size, batch_size, embed_dim)), 0, 1))
            
            negative_sample = 1 - torch.eye(batch_size, dtype=torch.long, device=anchor_dot_contrast.device)
            
            semantic_cosin_loss = 1 - anchor_dot_contrast.diag().sum() / batch_size
            semantic_cosin_loss += self.disentangler_negative_lambda + self.disentangler_negative_lambda * (anchor_dot_contrast * negative_sample).sum() / negative_sample.sum()
                        
            return semantic_cosin_loss
        
        def _language_cosin_loss(sample, reverse_sample, encoder_embedding_lang, reverse_encoder_embedding_lang):
            
            batch_size = encoder_embedding_lang.shape[0]
            feature_dim = encoder_embedding_lang.shape[1]
            anchor_feature = encoder_embedding_lang         # source
            contrast_feature = reverse_encoder_embedding_lang       # target
            
            similarity_function = self.similarity_function()

            anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                    torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))
            
            src_lang_tokens = sample["net_input"]["src_tokens"][: , 0]
            tgt_lang_tokens = reverse_sample["net_input"]["src_tokens"][: , 0]
            tgt_lang_tokens = tgt_lang_tokens.unsqueeze(dim=1)
            
            assert torch.all(src_lang_tokens > 32000).item()
            assert torch.all(tgt_lang_tokens > 32000).item()
            
            matrix_src = src_lang_tokens.expand((batch_size, batch_size))
            matrix_tgt = tgt_lang_tokens.expand((batch_size, batch_size))
            
            positive_sample = torch.where(matrix_src == matrix_tgt, 1, 0)
            negative_sample = 1 - positive_sample                    
            
            language_cosin_loss = 0
            if positive_sample.sum() > 0:
                language_cosin_loss += 1 - (anchor_dot_contrast * positive_sample).sum() / positive_sample.sum()
            if negative_sample.sum() > 0:
                language_cosin_loss += self.disentangler_negative_lambda + self.disentangler_negative_lambda * (anchor_dot_contrast * negative_sample).sum() / negative_sample.sum()
                        
            return language_cosin_loss
                
        # encoder_out shape: (src_len, batch, embed_dim)
        encoder_embedding = _sentence_embedding(encoder_out, sample)
        reverse_encoder_embedding = _sentence_embedding(reverse_encoder_out, reverse_sample)
        encoder_embedding_sem = _sentence_embedding(encoder_out_sem, sample)
        reverse_encoder_embedding_sem = _sentence_embedding(reverse_encoder_out_sem, reverse_sample)
        encoder_embedding_lang = _sentence_embedding(encoder_out_lang, sample)
        reverse_encoder_embedding_lang = _sentence_embedding(reverse_encoder_out_lang, reverse_sample)
        
        semantic_cosin_loss = _semantic_cosin_loss(encoder_embedding_sem, reverse_encoder_embedding_sem)
        language_cosin_loss = _language_cosin_loss(sample, reverse_sample, encoder_embedding_lang, reverse_encoder_embedding_lang)
        reconstruct_loss = _reconstruct_loss(encoder_embedding, reverse_encoder_embedding, encoder_embedding_sem, reverse_encoder_embedding_sem, encoder_embedding_lang, reverse_encoder_embedding_lang)
                
        det_loss = semantic_cosin_loss + language_cosin_loss + self.disentangler_reconstruction_lambda * reconstruct_loss
        
        return det_loss
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)

        batches = len(logging_outputs)
        same_lang_sentences = utils.item(
            sum(log.get("same_lang_sentences", 0) for log in logging_outputs)
        )
        
        det_loss = utils.item(
            sum(log.get("det_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "det_loss",
            det_loss / batches / math.log(2),
            batches,
            round=3,
        )
        
