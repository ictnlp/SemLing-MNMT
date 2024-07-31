# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.modules.quant_noise import quant_noise
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)
from fairseq.models.transformer.transformer_legacy import (
    TransformerModel,
)
from fairseq.models.transformer.transformer_encoder import (
    TransformerEncoderBase,
)
from fairseq.models.transformer.transformer_decoder import (
    TransformerDecoderBase,
)

# from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
# from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
# from fairseq.modules.quant_noise import quant_noise
# from torch import Tensor

# import pdb


@register_model("transformer_with_disentangler_and_linguistic_encoder")
class TransformerModelWithDisentanglerAndLinguisticEncoder(TransformerModel):

    @classmethod
    def hub_models(cls):
        # fmt: off
        return {
            'opus-100': 'https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz',
        }
        # fmt: on

    def __init__(self, args, encoder, disentangler, linguistic_encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.args = args
        self.disentangler = disentangler
        self.linguistic_encoder = linguistic_encoder

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=True, with_prefix=""
        )        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)
        
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        disentangler = cls.build_disentangler(args)
        linguistic_encoder = cls.build_linguistic_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder_with_semantic_and_linguistic_encoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, disentangler, linguistic_encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return super().build_embedding(
            args, dictionary, embed_dim, path
        )
        
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return super().build_encoder(
            args, src_dict, embed_tokens
        )
    
    @classmethod
    def build_disentangler(cls, args):
        return TransformerDisentangler(args)
    
    @classmethod
    def build_linguistic_encoder(cls, args, tgt_dict, embed_tokens):
        return TransformerLinguisticEncoder(args, tgt_dict, embed_tokens)
    
    @classmethod
    def build_decoder_with_semantic_and_linguistic_encoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderWithLinguisticEncoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        
        # encoder_out type: Encoderout, encoder_out.encoder_out shape: (src_len, batch, embed_dim)
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        
        encoder_out_sem, encoder_out_lang = self.disentangler(encoder_out)
        encoder_out["encoder_out"][0] = encoder_out_sem
        
        linguistic_encoder_out = self.linguistic_encoder(
            prev_output_tokens, src_lengths=prev_output_tokens.shape[1], return_all_hiddens=return_all_hiddens
        )

        # decoder_out type: (decoder_out, extra_feature), decoder_out[0] shape: (batch, tgt_len, vocab_dim)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            linguistic_encoder_out=linguistic_encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return decoder_out

class TransformerDisentangler(nn.Module):
    def __init__(self, args):
        
        super().__init__()
        
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        
        # activate layer in fully connected layer
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before
        
        # language FFN
        self.langfc1 = self.build_fc(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise_pq,
            self.quant_noise_block_size,
        )
        self.langfc2 = self.build_fc(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise_pq,
            self.quant_noise_block_size,
        )
        
        # semantic FFN
        self.semfc1 = self.build_fc(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise_pq,
            self.quant_noise_block_size,
        )
        self.semfc2 = self.build_fc(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise_pq,
            self.quant_noise_block_size,
        )
        
    def build_fc(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
    
    def forward(
        self,
        encoder_out: Optional[Dict[str, List[Tensor]]],
    ):
        
        # encoder_out (Tensor): the last encoder layer's output of shape `(src_len, batch, embed_dim)`
        enc: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
                
        # semantic of encoder_out
        enc_sem = self.activation_fn(self.semfc1(enc))
        enc_sem = self.activation_dropout_module(enc_sem)
        enc_sem = self.semfc2(enc_sem)
        enc_sem = self.dropout_module(enc_sem)
                
        # language of encoder_out
        enc_lang = self.activation_fn(self.langfc1(enc))
        enc_lang = self.activation_dropout_module(enc_lang)
        enc_lang = self.langfc2(enc_lang)
        enc_lang = self.dropout_module(enc_lang)

        return enc_sem, enc_lang
        
    
class TransformerLinguisticEncoder(TransformerEncoderBase):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(TransformerConfig.from_namespace(args), dictionary, embed_tokens, linguistic_encoder=True)
        self._future_mask = torch.empty(0)
    
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        # x.shape: (batch, src_len, embed_dim)
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        encoder_padding_mask_out = encoder_padding_mask if has_pads else None

        # encoder layers
        # layer: TransformerEncoderLayer()
        for layer in self.layers:
            
            self_attn_mask = self.buffered_future_mask(x)
                        
            lr = layer(x, encoder_padding_mask_out, self_attn_mask, speed_up=False)

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)
            
        # x.shape: (src_len, batch, embed_dim)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
                
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }
        # EncoderOut(
        #     encoder_out=x,  # T x B x C
        #     encoder_padding_mask=encoder_padding_mask,  # B x T
        #     encoder_embedding=encoder_embedding,  # B x T x C
        #     encoder_states=encoder_states,  # List[T x B x C]
        #     src_tokens=None,
        #     src_lengths=None,
        # )

class TransformerDecoderWithLinguisticEncoder(TransformerDecoderBase):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(TransformerConfig.from_namespace(args), dictionary, embed_tokens, no_encoder_attn=False)
        
        # ffn
        self.embed_dim = args.decoder_embed_dim
        self.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        
        # activate layer in fully connected layer
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before
        
        # fully connected layer
        self.fc1 = self.build_fc(
            self.embed_dim * 2,
            args.decoder_ffn_embed_dim,
            self.quant_noise_pq,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise_pq,
            self.quant_noise_block_size,
        )
        
    def build_fc(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)


    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn
        )

    # decoder forward
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        linguistic_encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
         
        # x shape: (batch, tgt_len, embed_dim)
        # linguistic_encoder_out.encoder_out shape: (tgt_len, batch, embed_dim)
        x = torch.cat((x, torch.transpose(linguistic_encoder_out["encoder_out"][0], 0, 1)), dim=-1)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        
        # linear, (batch, tgt_len, embed_dim) -> (batch, tgt_len, vocab_dim)
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return super().extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("transformer_with_disentangler_and_linguistic_encoder", "transformer_with_disentangler_and_linguistic_encoder")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    
    args.linguistic_encoder_embed_path = getattr(args, "linguistic_encoder_embed_path", None)
    args.linguistic_encoder_embed_dim = getattr(args, "linguistic_encoder_embed_dim", 512)
    args.linguistic_encoder_ffn_embed_dim = getattr(args, "linguistic_encoder_ffn_embed_dim", 2048)
    args.linguistic_encoder_layers = getattr(args, "linguistic_encoder_layers", 6)
    args.linguistic_encoder_attention_heads = getattr(args, "linguistic_encoder_attention_heads", 8)
    args.linguistic_encoder_normalize_before = getattr(args, "linguistic_encoder_normalize_before", False)
    args.linguistic_encoder_learned_pos = getattr(args, "linguistic_encoder_learned_pos", False)
    
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
