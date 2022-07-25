""" EsmEmbedding is a subnetwork of ProteinBert model that takes as input the first continuous embedding.
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


DEBUG=False


class EsmEmbedding(nn.Module):
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

	def __init__(self, original_model, alphabet):
		super().__init__()
		original_model.eval()

		self.original_model = original_model
		self.args = original_model.args
		self.alphabet = alphabet
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

		# start/end idxs of residues tokens in the alphabet
		self.start_token_idx, self.end_token_idx = 4, 29
		self.residues_tokens = self.alphabet.all_toks[self.start_token_idx:self.end_token_idx]

		self.check_correctness()

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

		for layer_idx, layer in enumerate(self.original_model.layers):
			x, attn = layer(
				x, self_attn_padding_mask=padding_mask, need_head_weights=need_head_weights
			)
			if (layer_idx + 1) in repr_layers:
				hidden_representations[layer_idx + 1] = x.transpose(0, 1)
			if need_head_weights:
				# (H, B, T, T) => (B, H, T, T)
				attn_weights.append(attn.transpose(1, 0))

		if self.original_model.model_version == "ESM-1b":
			x = self.original_model.emb_layer_norm_after(x)
			x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

			# last hidden representation should have layer norm applied
			if (layer_idx + 1) in repr_layers:
				hidden_representations[layer_idx + 1] = x
			x = self.original_model.lm_head(x)
		else:
			x = F.linear(x, self.original_model.embed_out, bias=self.original_model.embed_out_bias)
			x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

		result = {"logits": x, "representations": hidden_representations}
		if need_head_weights:
			# attentions: B x L x H x T x T
			attentions = torch.stack(attn_weights, 1)
			if self.original_model.model_version == "ESM-1":
				# ESM-1 models have an additional null-token for attention, which we remove
				attentions = attentions[..., :-1]
			if padding_mask is not None:
				attention_mask = 1 - padding_mask.type_as(attentions)
				attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
				attentions = attentions * attention_mask[:, None, None, :, :]
			result["attentions"] = attentions
			if return_contacts:
				contacts = self.original_model.contact_head(tokens, attentions)
				result["contacts"] = contacts

		return result

	def predict_contacts(self, first_embedding):
		return self(first_embedding, return_contacts=True)["contacts"]

	@property
	def num_layers(self):
		return self.args.layers

	def check_correctness(self, batch_tokens=None):
		""" check output logits are equal """

		if batch_tokens is None:
			data = [("test_sequence", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),]
			batch_converter = self.alphabet.get_batch_converter()
			_, _, batch_tokens = batch_converter(data)

		self.eval()
		self.original_model.eval()

		with torch.no_grad():

			device = next(self.original_model.parameters()).device
			batch_tokens = batch_tokens.to(device)
			self.to(device)

			results = self.original_model(batch_tokens.to(device), repr_layers=[0])
			orig_logits = results["logits"]

			first_embedding = results["representations"][0]
			emb_logits = self(first_embedding=first_embedding)["logits"]

			assert torch.all(torch.eq(orig_logits, emb_logits))

	def compute_tokens_attention(self, batch_tokens, layers_idxs):

		with torch.no_grad():
			results = self.original_model(batch_tokens, repr_layers=layers_idxs, return_contacts=True)

		attentions = results["attentions"]
		batch_size, n_layers, n_heads, n_tokens = attentions.shape[:4]

		if DEBUG:
			print(f"\nbatch_size = {batch_size}\tn_layers = {n_layers}\tn_heads = {n_heads}")

		assert batch_size==1 
		attentions = attentions[0, layers_idxs]

		### compute avg attention across all heads (1) and layers (0)
		avg_attentions = attentions.mean(1).mean(0).squeeze()
		assert avg_attentions.shape[0] == avg_attentions.shape[1]

		### remove start and end tokens attention
		tokens_attention = avg_attentions[1:-1, 1:-1]

		### compute l2 norm of attention vectors
		tokens_attention = torch.norm(tokens_attention, dim=-1, p=2)

		return tokens_attention

	def loss(self, method, output, target_token_idxs, true_residues_idxs):
		
		if method=='target_probs':
			logits = output['logits'][:,1:-1, :]
			probs = torch.softmax(logits, dim=-1)
			loss = torch.sum(probs[:,target_token_idxs])

		elif method=='tokens_repr':
			output_representations = output['representations'][self.args.layers][:].squeeze()
			output_representations = output_representations[1:-1, :]
			loss = torch.sum(torch.abs(output_representations[target_token_idxs,:]))

		elif method=='max_masked_prob':
			logits = output['logits'][:,1:-1, :]
			probs = torch.softmax(logits, dim=-1)

			target_probs = [probs[:,token_idx,residue_idx] 
				for token_idx, residue_idx in zip(target_token_idxs, true_residues_idxs)]

			loss = torch.max(torch.stack(target_probs))

		elif method=='sum_masked_prob':
			logits = output['logits'][:,1:-1, :]
			probs = torch.softmax(logits, dim=-1)

			target_probs = [probs[:,token_idx,residue_idx] 
				for token_idx, residue_idx in zip(target_token_idxs, true_residues_idxs)]

			loss = torch.sum(torch.stack(target_probs))

		else:
			raise AttributeError

		return loss

	def mask_batch_tokens(self, batch_tokens, target_token_idxs):

		for target_token_idx in target_token_idxs:
			batch_tokens[:, 1+target_token_idx] = self.alphabet.mask_idx

		return batch_tokens
