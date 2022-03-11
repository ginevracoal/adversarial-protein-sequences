""" EmbModel is a subnetwork of ProteinBert model that takes as input the first continuous embedding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from esm.modules import (
    TransformerLayer,
    AxialTransformerLayer,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    RobertaLMHead,
    ESM1bLayerNorm,
    ContactPredictionHead,
)

from esm.axial_attention import RowSelfAttention, ColumnSelfAttention


class EmbModel(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--num_layers", default=36, type=int, metavar="N", help="number of layers"
        )
        parser.add_argument(
            "--embed_dim", default=1280, type=int, metavar="N", help="embedding dimension"
        )
        parser.add_argument(
            "--logit_bias", action="store_true", help="whether to apply bias to logits"
        )
        parser.add_argument(
            "--ffn_embed_dim",
            default=5120,
            type=int,
            metavar="N",
            help="embedding dimension for FFN",
        )
        parser.add_argument(
            "--attention_heads",
            default=20,
            type=int,
            metavar="N",
            help="number of attention heads",
        )

    def __init__(self, bert, alphabet):
        super().__init__()
        self.bert = bert
        self.args = bert.args
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.emb_layer_norm_before = getattr(self.args, "emb_layer_norm_before", False)
        if self.args.arch == "roberta_large":
            self.model_version = "ESM-1b"
        else:
            self.model_version = "ESM-1"

    def _init_submodules_common(self):

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.args.embed_dim,
                    self.args.ffn_embed_dim,
                    self.args.attention_heads,
                    add_bias_kv=(self.model_version != "ESM-1b"),
                    use_esm1b_layer_norm=(self.model_version == "ESM-1b"),
                )
                for _ in range(self.args.layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.args.layers * self.args.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )

    def forward(self, first_embedding, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        x = first_embedding

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        # if not padding_mask.any():
        padding_mask = None # todo: check this on multiple inputs with different lengths

        for layer_idx, layer in enumerate(self.bert.layers):
            x, attn = layer(
                x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        if self.bert.model_version == "ESM-1b":
            x = self.bert.emb_layer_norm_after(x)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

            # last hidden representation should have layer norm applied
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x
            x = self.bert.lm_head(x)
        else:
            x = F.linear(x, self.bert.embed_out, bias=self.bert.embed_out_bias)
            x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if self.bert.model_version == "ESM-1":
                # ESM-1 models have an additional null-token for attention, which we remove
                attentions = attentions[..., :-1]
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.bert.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, first_embedding):
        return self(first_embedding, return_contacts=True)["contacts"]

    @property
    def num_layers(self):
        return self.args.layers

    def check_correctness(self, original_model, batch_tokens):
        """ check output logits are equal """
        
        with torch.no_grad():
            results = original_model(batch_tokens, repr_layers=[0])
            first_embedding = results["representations"][0]
            assert torch.all(torch.eq(self(first_embedding)['logits'], results['logits']))

# class SaliencyWrapper(nn.Module):
    
#     def __init__(self, model):
#         super(SaliencyWrapper, self).__init__()
#         self.model = model
        
#     def forward(self, embeddings):        
#         return self.model(embeddings)['logits']
