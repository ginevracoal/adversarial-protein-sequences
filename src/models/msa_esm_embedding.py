""" MsaEsmEmbedding is a subnetwork of MSATransformer model that takes as input the first continuous embedding.
"""

import math
import torch
import numpy as np
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


class MsaEsmEmbedding(nn.Module):
	@classmethod
	def add_args(cls, parser):
		# fmt: off
		parser.add_argument(
			"--num_layers",
			default=12,
			type=int,
			metavar="N",
			help="number of layers"
		)
		parser.add_argument(
			"--embed_dim",
			default=768,
			type=int,
			metavar="N",
			help="embedding dimension"
		)
		parser.add_argument(
			"--logit_bias",
			action="store_true",
			help="whether to apply bias to logits"
		)
		parser.add_argument(
			"--ffn_embed_dim",
			default=3072,
			type=int,
			metavar="N",
			help="embedding dimension for FFN",
		)
		parser.add_argument(
			"--attention_heads",
			default=12,
			type=int,
			metavar="N",
			help="number of attention heads",
		)
		parser.add_argument(
			"--dropout",
			default=0.1,
			type=float,
			help="Dropout to apply."
		)
		parser.add_argument(
			"--attention_dropout",
			default=0.1,
			type=float,
			help="Dropout to apply."
		)
		parser.add_argument(
			"--activation_dropout",
			default=0.1,
			type=float,
			help="Dropout to apply."
		)
		parser.add_argument(
			"--max_tokens_per_msa",
			default=2 ** 14,
			type=int,
			help=(
				"Used during inference to batch attention computations in a single "
				"forward pass. This allows increased input sizes with less memory."
			),
		)
		# fmt: on

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

		self.embed_tokens = nn.Embedding(
			self.alphabet_size, self.args.embed_dim, padding_idx=self.padding_idx
		)

		if getattr(self.args, "embed_positions_msa", False):
			emb_dim = getattr(self.args, "embed_positions_msa_dim", self.args.embed_dim)
			self.msa_position_embedding = nn.Parameter(
				0.01 * torch.randn(1, 1024, 1, emb_dim),
				requires_grad=True,
			)
		else:
			self.register_parameter("msa_position_embedding", None)

		self.dropout_module = nn.Dropout(self.args.dropout)
		self.layers = nn.ModuleList(
			[
				AxialTransformerLayer(
					self.args.embed_dim,
					self.args.ffn_embed_dim,
					self.args.attention_heads,
					self.args.dropout,
					self.args.attention_dropout,
					self.args.activation_dropout,
					getattr(self.args, "max_tokens_per_msa", self.args.max_tokens),
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
		self.embed_positions = LearnedPositionalEmbedding(
			self.args.max_positions,
			self.args.embed_dim,
			self.padding_idx,
		)
		self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
		self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
		self.lm_head = RobertaLMHead(
			embed_dim=self.args.embed_dim,
			output_dim=self.alphabet_size,
			weight=self.embed_tokens.weight,
		)

		# start/end idxs of residues tokens in the alphabet
		self.start_token_idx, self.end_token_idx = 4, 29
		self.residues_tokens = self.alphabet.all_toks[self.start_token_idx:self.end_token_idx]

		self.check_correctness()

	def forward(self, first_embedding, repr_layers=[], need_head_weights=False, return_contacts=False):

		if return_contacts:
			need_head_weights = True

		padding_mask = None
		x = first_embedding

		repr_layers = set(repr_layers)
		hidden_representations = {}
		if 0 in repr_layers:
			hidden_representations[0] = x

		if need_head_weights:
			row_attn_weights = []
			col_attn_weights = []

		# B x R x C x D -> R x C x B x D
		x = x.permute(1, 2, 0, 3)

		for layer_idx, layer in enumerate(self.original_model.layers):
			x = layer(
				x,
				self_attn_padding_mask=padding_mask,
				need_head_weights=need_head_weights,
			)
			if need_head_weights:
				x, col_attn, row_attn = x
				# H x C x B x R x R -> B x H x C x R x R
				col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4))
				# H x B x C x C -> B x H x C x C
				row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
			if (layer_idx + 1) in repr_layers:
				hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)

		x = self.original_model.emb_layer_norm_after(x)
		x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D

		# last hidden representation should have layer norm applied
		if (layer_idx + 1) in repr_layers:
			hidden_representations[layer_idx + 1] = x
		x = self.original_model.lm_head(x)

		result = {"logits": x, "representations": hidden_representations}
		if need_head_weights:
			# col_attentions: B x L x H x C x R x R
			col_attentions = torch.stack(col_attn_weights, 1)
			# row_attentions: B x L x H x C x C
			row_attentions = torch.stack(row_attn_weights, 1)
			result["col_attentions"] = col_attentions
			result["row_attentions"] = row_attentions
			if return_contacts:
				contacts = self.original_model.contact_head(tokens, row_attentions)
				result["contacts"] = contacts

		return result

	def predict_contacts(self, tokens):
		return self(tokens, return_contacts=True)["contacts"]

	@property
	def num_layers(self):
		return self.args.layers

	def max_tokens_per_msa_(self, value: int) -> None:
		"""The MSA Transformer automatically batches attention computations when
		gradients are disabled to allow you to pass in larger MSAs at test time than
		you can fit in GPU memory. By default this occurs when more than 2^14 tokens
		are passed in the input MSA. You can set this value to infinity to disable
		this behavior.
		"""
		for module in self.modules():
			if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
				module.max_tokens_per_msa = value

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

			results = self.original_model(batch_tokens, repr_layers=[0])
			orig_logits = results["logits"]

			first_embedding = results["representations"][0]
			emb_logits = self(first_embedding=first_embedding)["logits"]

			assert torch.all(torch.eq(orig_logits, emb_logits))

	def compute_tokens_attention(self, batch_tokens, layers_idxs, gamma=0.5):

		with torch.no_grad():
			results = self.original_model(batch_tokens, repr_layers=layers_idxs, return_contacts=True)
		
		row_attentions = results["row_attentions"]
		col_attentions = results["col_attentions"]
		batch_size = col_attentions.shape[-1]
		_, n_layers, n_heads, n_tokens = row_attentions.shape[:4]

		if DEBUG:
			print(f"\nbatch_size = {batch_size}\tn_layers = {n_layers}\tn_heads = {n_heads}")
			print("\nrow_attentions.shape =", row_attentions.shape)
			print("col_attentions.shape =", col_attentions.shape)

		row_attentions = row_attentions[0, layers_idxs]
		col_attentions = col_attentions[0, layers_idxs]

		### compute avg attention across all heads (1) and layers (0)
		row_attentions = row_attentions.mean(1).mean(0).squeeze()
		col_attentions = col_attentions.mean(1).mean(0).squeeze()
		assert row_attentions.shape[0] == row_attentions.shape[1]
		assert col_attentions.shape[1] == col_attentions.shape[2]

		### remove start tokens attention
		row_attentions = row_attentions[1:, 1:]
		col_attentions = col_attentions[1:].flatten(start_dim=1)

		### compute l2 norm of attention vectors
		row_attentions = torch.norm(row_attentions, dim=-1, p=2)
		col_attentions = torch.norm(col_attentions, dim=-1, p=2)
		tokens_attention = gamma*F.softmax(row_attentions, dim=-1) + (1-gamma)*F.softmax(col_attentions, dim=-1)

		return tokens_attention

	def loss(self, method, output, target_token_idxs, true_residues_idxs):

		if method=='target_probs':
			logits = output['logits'][:,0,1:, :]
			probs = torch.softmax(logits, dim=-1)
			loss = torch.sum(probs[:,target_token_idxs])

		elif method=='tokens_repr':
			output_representations = output['representations'][self.args.layers][:,0].squeeze()
			output_representations = output_representations[1:, :]
			loss = torch.sum(torch.abs(output_representations[target_token_idxs,:]))

		elif method=='max_masked_prob':
			logits = output['logits'][:,0,1:, :]
			probs = torch.softmax(logits, dim=-1)

			target_probs = [probs[:,token_idx,residue_idx] 
				for token_idx, residue_idx in zip(target_token_idxs, true_residues_idxs)]

			loss = torch.max(torch.stack(target_probs))

		else:
			raise AttributeError

		return loss

	def mask_batch_tokens(self, batch_tokens, target_token_idxs):
		""" Mask the first sequence in the batch at `target_token_idx`.
		"""
		assert batch_tokens.shape[1]>1

		for target_token_idx in target_token_idxs:
			batch_tokens[:, 0, 1+target_token_idx] = self.alphabet.mask_idx

		return batch_tokens

	def compute_tokens_entropy(self, msa):

		msa_array = np.array([list(sequence) for sequence in dict(msa).values()])

		n_residues = len(self.residues_tokens)
		n_sequences = msa_array.shape[0]
		n_tokens = msa_array.shape[1]

		### count occurrence probs of residues in msa columns

		occurrence_frequencies = torch.empty((n_tokens, n_residues))

		for residue_idx, residue in enumerate(self.residues_tokens):
			for token_idx in range(n_tokens):
				column_string = "".join(msa_array[:,token_idx])
				occurrence_frequencies[token_idx,residue_idx] = column_string.count(residue)

		occurrence_probs = torch.softmax(occurrence_frequencies, dim=1)

		### compute token idxs entropy

		tokens_entropy = torch.tensor([torch.sum(torch.tensor([-p_ij*torch.log(p_ij) for p_ij in occurrence_probs[i,:]])) 
			for i in range(n_tokens)])

		return tokens_entropy

	def get_frequencies(self, msa):

		msa_array = np.array([list(sequence) for sequence in dict(msa).values()])

		n_residues = len(self.alphabet.all_toks)
		n_sequences = msa_array.shape[0]
		n_tokens = msa_array.shape[1]

		### count occurrence probs of residues in msa columns

		occurrence_frequencies = torch.empty((n_tokens, n_residues))

		for residue_idx, residue in enumerate(self.alphabet.all_toks):
			for token_idx in range(n_tokens):
				column_string = "".join(msa_array[:,token_idx])
				occurrence_frequencies[token_idx,residue_idx] = column_string.count(residue)

		occurrence_probs = torch.softmax(occurrence_frequencies, dim=1)

		return occurrence_probs

