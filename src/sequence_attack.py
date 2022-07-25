import torch
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from Bio.SubsMat import MatrixInfo
from utils.protein_sequences import *

blosum = MatrixInfo.blosum62
DEBUG=True


class SequenceAttack():

	def __init__(self, original_model, embedding_model, alphabet):
		original_model.eval()
		embedding_model.eval()

		self.original_model = original_model
		self.embedding_model = embedding_model
		self.alphabet = alphabet

		self.start_token_idx = self.embedding_model.start_token_idx
		self.end_token_idx = self.embedding_model.end_token_idx
		self.residues_tokens = self.embedding_model.residues_tokens

	def _get_attention_layers_idxs(self, target_attention):

		n_layers = self.original_model.args.layers

		if target_attention=='all_layers':
			return list(range(n_layers))

		elif target_attention=='last_layer':
			return [n_layers-1]

		else:
			raise AttributeError

	def choose_target_token_idxs(self, batch_tokens, n_token_substitutions, token_selection='max_attention', 
		target_attention='last_layer', msa=None, verbose=False):

		if verbose:
			print("\n-- Choosing target token idxs --")

		layers_idxs = self._get_attention_layers_idxs(target_attention)
		tokens_attention = self.embedding_model.compute_tokens_attention(batch_tokens=batch_tokens, layers_idxs=layers_idxs)

		if token_selection=='max_attention':

			target_tokens_attention, target_token_idxs = torch.topk(tokens_attention, k=n_token_substitutions, largest=True)

		elif token_selection=='min_entropy':

			if msa is None:
				raise AttributeError("Entropy selection needs an MSA")

			tokens_entropy = self.embedding_model.compute_tokens_entropy(msa=msa)
			target_tokens_entropy, target_token_idxs = torch.topk(tokens_entropy, k=n_token_substitutions, largest=False)
			target_tokens_attention = tokens_attention[target_token_idxs]

		elif token_selection=='max_entropy':

			if msa is None:
				raise AttributeError("Entropy selection needs an MSA")

			tokens_entropy = self.embedding_model.compute_tokens_entropy(msa=msa)
			target_tokens_entropy, target_token_idxs = torch.topk(tokens_entropy, k=n_token_substitutions, largest=True)
			target_tokens_attention = tokens_attention[target_token_idxs]

		else:
			raise AttributeError("Wrong token_selection method")

		if verbose:
			print(f"\ntarget_token_idxs = {target_token_idxs}")

		return target_token_idxs.cpu().detach().numpy(), target_tokens_attention.cpu().detach().numpy()

	# def compute_tokens_entropy(self, msa, n_token_substitutions):

	# 	msa_array = np.array([list(sequence) for sequence in dict(msa).values()])

	# 	n_residues = len(self.residues_tokens)
	# 	n_sequences = msa_array.shape[0]
	# 	n_tokens = msa_array.shape[1]

	# 	### count occurrence probs of residues in msa columns

	# 	occurrence_frequencies = torch.empty((n_tokens, n_residues))

	# 	for residue_idx, residue in enumerate(self.residues_tokens):
	# 		for token_idx in range(n_tokens):
	# 			column_string = "".join(msa_array[:,token_idx])
	# 			occurrence_frequencies[token_idx,residue_idx] = column_string.count(residue)

	# 	occurrence_probs = torch.softmax(occurrence_frequencies, dim=1)

	# 	### compute token idxs entropy

	# 	tokens_entropy = torch.tensor([torch.sum(torch.tensor([-p_ij*torch.log(p_ij) for p_ij in occurrence_probs[i,:]])) 
	# 		for i in range(n_tokens)])

	# 	return tokens_entropy

	def compute_loss_gradient(self, original_sequence, batch_tokens, target_token_idxs, first_embedding, loss_method, 
		verbose=False):

		if verbose:
			print("\n= Computing loss gradients =")

		if loss_method=='max_masked_prob':

			with torch.no_grad():
				device = next(self.original_model.parameters()).device
				n_layers = self.original_model.args.layers

				batch_tokens_masked = self.embedding_model.mask_batch_tokens(batch_tokens, target_token_idxs=target_token_idxs)
				results = self.original_model(batch_tokens_masked, repr_layers=[0], return_contacts=True)
				first_embedding = results["representations"][0].to(device)

		first_embedding.requires_grad=True

		true_residues_idxs = [self.alphabet.get_idx(original_sequence[token_idx]) for token_idx in target_token_idxs]

		output = self.embedding_model(first_embedding=first_embedding, repr_layers=[self.original_model.args.layers])

		loss = self.embedding_model.loss(method=loss_method, output=output, 
			target_token_idxs=target_token_idxs, true_residues_idxs=true_residues_idxs)

		if DEBUG:
			print(f"\n{loss_method} loss =", loss.item())

		self.embedding_model.zero_grad()
		loss.backward()

		signed_gradient = first_embedding.grad.data.sign()
		first_embedding.requires_grad=False
		signed_gradient.requires_grad=False

		return signed_gradient, loss

	def get_allowed_token_substitutions(self, current_token, blosum_check=True):
		""" Get the list of allowed substitutions of `current_token` with new residues from the alphabet.
		If `blosum_check` is True, remove substitutions with null frequency in BLOSUM62 matrix. 
		If there are no allowed substitutions keep the original token fixed. 
		"""

		allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token,'X']))
		
		if blosum_check:
			blosum = MatrixInfo.blosum62

			for new_token in allowed_token_substitutions:
				if ((current_token, new_token) not in blosum.keys()) & ((new_token, current_token) not in blosum.keys()):
					allowed_token_substitutions = list(set(allowed_token_substitutions) - set([new_token]))

		if len(allowed_token_substitutions)==0:
			print("\nNo substitutions allowed, keeping the original token fixed.")
			allowed_token_substitutions = [current_token]

		return allowed_token_substitutions

	# def attack(self, name, original_sequence, original_batch_tokens, target_token_idxs, target_tokens_attention,
	# 	first_embedding, signed_gradient, msa=None, verbose=False,
	# 	perturbations_keys=['masked_pred','max_cos','min_dist','max_dist'], p_norm=1):
	# 	""" Compute perturbations of the original sequence at target token idxs based on the chosen perturbation methods.
	# 	Evaluate perturbed sequences against the original one.
	# 	"""

	# 	self.original_model.eval()
	# 	self.embedding_model.eval()

	# 	if verbose:
	# 		print("\n-- Building adversarial sequences --")

	# 	adv_perturbations_keys = perturbations_keys.copy()

	# 	assert 'masked_pred' in perturbations_keys
	# 	adv_perturbations_keys.remove('masked_pred')
		
	# 	if msa: 
	# 		first_embedding=first_embedding[:,0]
	# 		signed_gradient=signed_gradient[:,0]

	# 	batch_converter = self.embedding_model.alphabet.get_batch_converter()

	# 	### init dictionary
	# 	atk_dict = {
	# 		'name':name,
	# 		'original_sequence':original_sequence,
	# 		'orig_tokens':[],
	# 		'target_token_idxs':target_token_idxs,
	# 		'masked_pred_accuracy':0.}

	# 	for pert_key in perturbations_keys:
	# 		atk_dict.update({
	# 			f'{pert_key}_tokens':[], 
	# 			f'{pert_key}_sequence':original_sequence,
	# 			f'{pert_key}_embedding_distance':0.,
	# 			f'{pert_key}_pseudo_likelihood':0.,
	# 			f'{pert_key}_evo_velocity':0.,
	# 			f'{pert_key}_blosum_dist':0.
	# 			})

	# 	atk_dict['orig_tokens'] = [list(original_sequence)[target_token_idx] for target_token_idx in target_token_idxs]

	# 	### build dictionary of allowed tokens substitutions for each target token idx

	# 	tokens_substitutions_dict = {}
	# 	for target_token_idx in target_token_idxs:
	# 		current_token = str(list(original_sequence)[target_token_idx])
	# 		allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))

	# 		tokens_substitutions_dict[f'token_{str(target_token_idx)}'] = {'current_token':current_token, 
	# 			'allowed_token_substitutions':allowed_token_substitutions}

	# 	allowed_token_substitutions_list = [tokens_substitutions_dict[f'token_{str(target_token_idx)}']['allowed_token_substitutions'] \
	# 		for target_token_idx in target_token_idxs]
	# 	allowed_seq_substitutions = list(itertools.product(*allowed_token_substitutions_list))
		
	# 	### build dict of all possible perturbed sequences

	# 	perturbed_sequences_dict = {}
	# 	for i, sequence_substitution in enumerate(allowed_seq_substitutions):
	# 		original_sequence_list = list(original_sequence)

	# 		for target_token_idx, new_token in zip(target_token_idxs, sequence_substitution):
	# 			current_token = original_sequence_list[target_token_idx]
	# 			original_sequence_list[target_token_idx] = new_token

	# 		new_sequence = "".join(original_sequence_list)
	# 		perturbed_sequences_dict[f'seq_{str(i)}'] = {'new_tokens':sequence_substitution,'perturbed_sequence':new_sequence}

	# 	if verbose:
	# 		print("\ntokens_substitutions_dict:\n", tokens_substitutions_dict)
	# 		print("\nn. perturbed_sequences =", len(perturbed_sequences_dict))
	# 		print("\nperturbed_sequences_dict first item:\n", list(perturbed_sequences_dict.items())[0])

	# 	### mask original sequence at target_token_idx

	# 	batch_tokens_masked = self.embedding_model.mask_batch_tokens(original_batch_tokens.clone(), 
	# 		target_token_idxs=target_token_idxs)

	# 	### compute sequence attacks

	# 	embeddings_distances = []

	# 	atk_dict['max_cos'] = -1
	# 	atk_dict['min_dist'] = 10e10
	# 	atk_dict['max_dist'] = 0

	# 	if DEBUG:
	# 		print(f"\ncurrent tokens at positions {target_token_idxs} = {atk_dict['orig_tokens']}")

	# 	for sequence_substitution in tqdm(perturbed_sequences_dict.values()):
	# 		perturbed_sequence = sequence_substitution['perturbed_sequence']

	# 		for pert_key in adv_perturbations_keys:

	# 			with torch.no_grad():

	# 				if msa:
	# 					perturbed_batch = [("pert_seq", perturbed_sequence)] + list(msa[1:])
	# 					batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)
	# 					results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
	# 					z_c = results["representations"][0][:,0]

	# 				else:
	# 					perturbed_batch = [("pert_seq", perturbed_sequence)]
	# 					batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)
	# 					results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
	# 					z_c = results["representations"][0]

	# 				z_c_diff = z_c-first_embedding
	# 				print(z_c_diff)
	# 				print(signed_gradient)
	# 				cosine_similarity = nn.CosineSimilarity(dim=-1)(signed_gradient.flatten(), z_c_diff.flatten())
	# 				embedding_distance = torch.norm(z_c_diff, p=p_norm)
	# 				embeddings_distances.append(embedding_distance)

	# 				### substitutions that maximize cosine similarity w.r.t. gradient direction

	# 				if pert_key=='max_cos' and cosine_similarity > atk_dict[pert_key]:

	# 					atk_dict[pert_key] = cosine_similarity.item()
	# 					atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
	# 					atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
	# 					atk_dict[f'{pert_key}_tokens'] = sequence_substitution['new_tokens']

	# 					if DEBUG:
	# 						print(f"\n\tnew tokens = {atk_dict[f'{pert_key}_tokens']}\tmax_cos_similarity = {cosine_similarity}")

	# 				### substitutions that minimize/maximize distance from the original embedding

	# 				if pert_key=='min_dist' and embedding_distance < atk_dict[pert_key]:
	# 					atk_dict[pert_key] = embedding_distance.item()
	# 					atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
	# 					atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
	# 					atk_dict[f'{pert_key}_tokens'] = sequence_substitution['new_tokens']

	# 					if DEBUG:
	# 						print(f"\n\tnew tokens = {atk_dict[f'{pert_key}_tokens']}\tmin_distance = {embedding_distance}")

	# 				if pert_key=='max_dist' and embedding_distance > atk_dict[pert_key]:
	# 					atk_dict[pert_key] = embedding_distance.item()
	# 					atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
	# 					atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
	# 					atk_dict[f'{pert_key}_tokens'] = sequence_substitution['new_tokens']

	# 					if DEBUG:
	# 						print(f"\n\tnew tokens = {atk_dict[f'{pert_key}_tokens']}\tmax_distance = {embedding_distance}")
		
	# 	### prediction on sequence masked at target_token_idxs

	# 	masked_prediction = self.original_model(batch_tokens_masked.to(signed_gradient.device))
	# 	predicted_sequence_list = list(original_sequence)

	# 	for i, target_token_idx in enumerate(target_token_idxs):
			
	# 		if msa:
	# 			logits = masked_prediction["logits"][:,0].squeeze()
	# 		else:
	# 			logits = masked_prediction["logits"].squeeze()

	# 		assert len(logits.shape)==2
	# 		logits = logits[1:len(original_sequence)+1, :]
	# 		probs = torch.softmax(logits, dim=-1)

	# 		predicted_residue_idx = probs[target_token_idx, :].argmax()
	# 		predicted_token = self.alphabet.all_toks[predicted_residue_idx]
	# 		predicted_sequence_list[target_token_idx] = predicted_token
	# 		atk_dict['masked_pred_tokens'].append(predicted_token)

	# 		atk_dict['masked_pred_accuracy'] += int(predicted_token==original_sequence[target_token_idx])/len(target_token_idxs)
			
	# 		if DEBUG:
	# 			print("\n\tpert_key = masked_pred")
	# 			print(f"\t\tpred_token = {predicted_token}, true_token = {original_sequence[target_token_idx]}, masked_pred_acc = {atk_dict['masked_pred_accuracy']}")

	# 		### compute confidence scores

	# 		for pert_key in perturbations_keys:

	# 			orig_token = original_sequence[target_token_idx]
	# 			new_token = atk_dict[f'{pert_key}_tokens'][i]

	# 			orig_residue_idx = self.alphabet.get_idx(orig_token)
	# 			new_residue_idx = self.alphabet.get_idx(new_token)

	# 			orig_log_prob = torch.log(probs[target_token_idx, orig_residue_idx])
	# 			adv_prob = probs[target_token_idx, new_residue_idx]
	# 			adv_log_prob = torch.log(adv_prob)

	# 			atk_dict[f'{pert_key}_pseudo_likelihood'] += (adv_prob/len(target_token_idxs)).item()
	# 			atk_dict[f'{pert_key}_evo_velocity'] += ((adv_log_prob-orig_log_prob)/len(target_token_idxs)).item()

	# 			if atk_dict[f'orig_tokens']==atk_dict[f'{pert_key}_tokens']:
	# 				assert atk_dict[f'{pert_key}_evo_velocity']==0.

	# 	predicted_sequence = "".join(predicted_sequence_list)

	# 	if msa:
	# 		predicted_batch = [("pred_seq", predicted_sequence)] + list(msa[1:])
	# 	else:
	# 		predicted_batch = [("pred_seq", predicted_sequence)]

	# 	batch_labels, batch_strs, batch_tokens = batch_converter(predicted_batch)
	# 	results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
	# 	z_c = results["representations"][0]

	# 	if msa:
	# 		z_c = z_c[:,0]

	# 	embedding_distance = torch.norm(first_embedding-z_c, p=p_norm)
	# 	atk_dict[f'masked_pred_sequence'] = predicted_sequence
	# 	atk_dict[f'masked_pred_embedding_distance'] = embedding_distance.item()

	# 	### compute blosum distances

	# 	if verbose:
	# 		print(f"\nSequence perturbation at target_token_idxs = {target_token_idxs}:")
	# 		print(f"\norig_tokens = {atk_dict['orig_tokens']}\tmasked_pred_accuracy = {atk_dict['masked_pred_accuracy']}")

	# 	for pert_key in perturbations_keys:
	# 		atk_dict[f'{pert_key}_blosum_dist'] = compute_blosum_distance(original_sequence, 
	# 			atk_dict[f'{pert_key}_sequence'], target_token_idxs)

	# 		if verbose:
	# 			print(f"\n{pert_key}\t", end="\t")
	# 			for dict_key in ['tokens','pseudo_likelihood','evo_velocity','blosum_dist']:
	# 				print(f"{dict_key} = {atk_dict[f'{pert_key}_{dict_key}']}", end="\t")

	# 	### unstack tokens lists

	# 	atk_df = pd.DataFrame()
	# 	for i, token_idx in enumerate(target_token_idxs):
	# 		row = atk_dict.copy()

	# 		token_idx = row['target_token_idxs'][i]
	# 		row['target_token_idx'] = row.pop('target_token_idxs')
	# 		row['target_token_idx'] = token_idx
	# 		row['target_token_attention'] = target_tokens_attention[i]

	# 		token = row['orig_tokens'][i]
	# 		row['orig_token'] = row.pop('orig_tokens')
	# 		row['orig_token'] = token

	# 		for pert_key in perturbations_keys:
	# 			token = row[f'{pert_key}_tokens'][i]
	# 			row[f'{pert_key}_token'] = row.pop(f'{pert_key}_tokens')
	# 			row[f'{pert_key}_token'] = token

	# 		atk_df = atk_df.append(row, ignore_index=True)

	# 	for key in perturbations_keys:
	# 		assert len(atk_df[f'{key}_sequence'].unique())==1

	# 	assert len(atk_df)==len(target_token_idxs)
	# 	return atk_df, torch.tensor(embeddings_distances)

	def incremental_attack(self, name, original_sequence, original_batch_tokens, target_token_idxs, target_tokens_attention,
		first_embedding, signed_gradient, msa=None, verbose=False,
		perturbations_keys=['masked_pred','max_cos','min_dist','max_dist'], p_norm=1):
		""" Compute perturbations of the original sequence at target token idxs based on the chosen perturbation methods.
		Evaluate perturbed sequences against the original one.
		"""

		self.original_model.eval()
		self.embedding_model.eval()

		if verbose:
			print("\n-- Building adversarial sequences --")

		adv_perturbations_keys = perturbations_keys.copy()

		assert 'masked_pred' in perturbations_keys
		adv_perturbations_keys.remove('masked_pred')
		
		batch_converter = self.embedding_model.alphabet.get_batch_converter()
		batch_tokens_masked = original_batch_tokens.clone()

		### init dictionary
		atk_dict = {
			'name':name,
			'original_sequence':original_sequence,
			'original_tokens':[],
			'target_token_idxs':target_token_idxs,
			'masked_pred_accuracy':0.}

		for pert_key in perturbations_keys:
			atk_dict.update({
				f'{pert_key}_tokens':[], 
				f'{pert_key}_sequence':original_sequence,
				f'{pert_key}_embedding_distance':0.,
				f'{pert_key}_pseudo_likelihood':0.,
				f'{pert_key}_evo_velocity':0.,
				f'{pert_key}_blosum_dist':0.
				})

		if msa is not None:
			first_embedding=first_embedding[:,0]
			signed_gradient=signed_gradient[:,0]
			msa_frequencies = self.embedding_model.get_frequencies(msa=msa)

		embeddings_distances = []
		
		for target_token_idx in target_token_idxs:

			original_token = original_sequence[target_token_idx]
			atk_dict['original_tokens'].append(original_token)

			### mask original sequence at target_token_idx
			batch_tokens_masked = self.embedding_model.mask_batch_tokens(batch_tokens_masked, 
				target_token_idxs=[target_token_idx])

			### allowed substitutions at target_token_idx 
			allowed_token_substitutions = self.get_allowed_token_substitutions(original_token)

			if DEBUG:
				print(f"\nallowed_token_substitutions at idx {target_token_idx} = {allowed_token_substitutions}")

			for pert_key in adv_perturbations_keys:

				if DEBUG:
					print("\n\tpert_key =", pert_key)
					print(f"\t\toriginal token = {original_token}")

				if pert_key=='max_cmap_dist':
					atk_dict.update({pert_key:0})

				if pert_key=='max_cos':
					atk_dict.update({pert_key:-1})

				if pert_key=='min_dist':
					atk_dict.update({pert_key:10e10})

				if pert_key=='max_dist':
					atk_dict.update({pert_key:0})

				if pert_key=='max_entropy' and msa is not None:
					atk_dict.update({pert_key:0})

				### updating one token at a time
				for j, token in enumerate(allowed_token_substitutions):
					current_sequence_list = list(atk_dict[f'{pert_key}_sequence'])
					current_sequence_list[target_token_idx] = token
					perturbed_sequence = "".join(current_sequence_list)

					with torch.no_grad():

						if msa is None: # single-seq attack
							perturbed_batch = [("pert_seq", perturbed_sequence)]
							batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)

							results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
							zc = results["representations"][0]

							assert zc.shape[1]==len(original_sequence)+2
							assert first_embedding.shape[1]==len(original_sequence)+2

						else: # msa attack
							perturbed_batch = [("pert_seq", perturbed_sequence)] + list(msa[1:])
							batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)
							results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
							zc = results["representations"][0][:,0]

							assert zc.shape[1]==len(original_sequence)+1
							assert first_embedding.shape[1]==len(original_sequence)+1

						zc_diff = zc-first_embedding

						n_diff_components = len(torch.nonzero(zc_diff.flatten()))
						assert n_diff_components > 0

						cosine_similarity = nn.CosineSimilarity(dim=-1)(signed_gradient.flatten(), zc_diff.flatten())
						embedding_distance = torch.norm(zc_diff, p=p_norm)
						embeddings_distances.append(embedding_distance)

						cmaps_distance = compute_cmaps_distance(model=self.original_model, alphabet=self.alphabet, 
							original_sequence=original_sequence, sequence_name=name, 
							perturbed_sequence=perturbed_sequence)

						if msa is not None:
							residue_idx = self.alphabet.get_idx(token)
							p = msa_frequencies[target_token_idx,residue_idx]
							token_entropy = -p*torch.log(p)

						### substitutions that maximize the distance bw original and perturbed contact maps

						if pert_key=='max_cmap_dist' and cmaps_distance > atk_dict[pert_key]:
							atk_dict[pert_key] = cmaps_distance
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token
							
							if DEBUG:
								print(f"\t\tnew token = {new_token}\tcmaps_distance = {cmaps_distance}")

						### substitutions that maximize cosine similarity w.r.t. gradient direction

						if pert_key=='max_cos' and cosine_similarity > atk_dict[pert_key]:
							atk_dict[pert_key] = cosine_similarity.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token
							
							if DEBUG:
								print(f"\t\tnew token = {new_token}\tcos_similarity = {cosine_similarity}")

						### substitutions that minimize/maximize distance from the original embedding

						if pert_key=='min_dist' and embedding_distance < atk_dict[pert_key]:
							atk_dict[pert_key] = embedding_distance.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token

							if DEBUG:
								print(f"\t\tnew token = {new_token}\tdistance = {embedding_distance}")

						if pert_key=='max_dist' and embedding_distance > atk_dict[pert_key]:
							atk_dict[pert_key] = embedding_distance.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token

							if DEBUG:
								print(f"\t\tnew token = {new_token}\tdistance = {embedding_distance}")

						if pert_key=='max_entropy' and token_entropy > atk_dict[pert_key]:
							atk_dict[pert_key] = token_entropy.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token

							if DEBUG:
								print(f"\t\tnew token = {new_token}\tentropy = {token_entropy}")

				atk_dict[f'{pert_key}_tokens'].append(new_token)

		assert len(atk_dict[f'{pert_key}_tokens'])==len(target_token_idxs)
		
		### prediction on sequence masked at target_token_idxs

		masked_prediction = self.original_model(batch_tokens_masked.to(signed_gradient.device))
		predicted_sequence_list = list(original_sequence)

		print("\n\tpert_key = masked_pred")

		for i, target_token_idx in enumerate(target_token_idxs):
			
			if msa:
				logits = masked_prediction["logits"][:,0].squeeze()
			else:
				logits = masked_prediction["logits"].squeeze()

			assert len(logits.shape)==2
			logits = logits[1:len(original_sequence)+1, :]
			probs = torch.softmax(logits, dim=-1)

			predicted_residue_idx = probs[target_token_idx, :].argmax()
			predicted_token = self.alphabet.all_toks[predicted_residue_idx]
			predicted_sequence_list[target_token_idx] = predicted_token
			atk_dict['masked_pred_tokens'].append(predicted_token)

			atk_dict['masked_pred_accuracy'] += int(predicted_token==original_sequence[target_token_idx])/len(target_token_idxs)
			
			if DEBUG:
				print(f"\t\tpred_token = {predicted_token}, true_token = {original_sequence[target_token_idx]}, masked_pred_acc = {atk_dict['masked_pred_accuracy']}")

			### compute confidence scores

			for pert_key in perturbations_keys:

				original_token = original_sequence[target_token_idx]
				new_token = atk_dict[f'{pert_key}_tokens'][i]

				original_residue_idx = self.alphabet.get_idx(original_token)
				new_residue_idx = self.alphabet.get_idx(new_token)

				original_log_prob = torch.log(probs[target_token_idx, original_residue_idx])
				adv_prob = probs[target_token_idx, new_residue_idx]
				adv_log_prob = torch.log(adv_prob)

				atk_dict[f'{pert_key}_pseudo_likelihood'] += (adv_prob/len(target_token_idxs)).item()
				atk_dict[f'{pert_key}_evo_velocity'] += ((adv_log_prob-original_log_prob)/len(target_token_idxs)).item()

				if atk_dict[f'original_tokens']==atk_dict[f'{pert_key}_tokens']:
					assert atk_dict[f'{pert_key}_evo_velocity']==0.

		predicted_sequence = "".join(predicted_sequence_list)

		if msa:
			predicted_batch = [("pred_seq", predicted_sequence)] + list(msa[1:])
		else:
			predicted_batch = [("pred_seq", predicted_sequence)]

		batch_labels, batch_strs, batch_tokens = batch_converter(predicted_batch)
		results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
		z_c = results["representations"][0]

		if msa:
			z_c = z_c[:,0]

		embedding_distance = torch.norm(first_embedding-z_c, p=p_norm)
		atk_dict[f'masked_pred_sequence'] = predicted_sequence
		atk_dict[f'masked_pred_embedding_distance'] = embedding_distance.item()

		### compute blosum distances

		if verbose:
			print(f"\nSequence perturbation at target_token_idxs = {target_token_idxs}:")
			print(f"\noriginal_tokens = {atk_dict['original_tokens']}\tmasked_pred_accuracy = {atk_dict['masked_pred_accuracy']}")

		for pert_key in perturbations_keys:
			atk_dict[f'{pert_key}_blosum_dist'] = compute_blosum_distance(original_sequence, 
				atk_dict[f'{pert_key}_sequence'], target_token_idxs)

			if verbose:
				print(f"\n{pert_key}\t", end="\t")
				for dict_key in ['pseudo_likelihood','evo_velocity','blosum_dist']: # 'tokens'
					print(f"{dict_key} = {atk_dict[f'{pert_key}_{dict_key}']}", end="\t")

		### unstack tokens lists

		atk_df = pd.DataFrame()
		for i, token_idx in enumerate(target_token_idxs):
			row = atk_dict.copy()

			token_idx = row['target_token_idxs'][i]
			row['target_token_idx'] = row.pop('target_token_idxs')
			row['target_token_idx'] = token_idx
			row['target_token_attention'] = target_tokens_attention[i]

			token = row['original_tokens'][i]
			row['original_token'] = row.pop('original_tokens')
			row['original_token'] = token

			for pert_key in perturbations_keys:
				token = row[f'{pert_key}_tokens'][i]
				row[f'{pert_key}_token'] = row.pop(f'{pert_key}_tokens')
				row[f'{pert_key}_token'] = token

			atk_df = atk_df.append(row, ignore_index=True)


		# if DEBUG:

		# 	print("\n\nCheck blosum scores of single substitutions:")
		# 	for pert_key in perturbations_keys:
		# 		print(f"\n{pert_key} blosum scores: ", end="\t")

		# 		for _, atk_row in atk_df.iterrows():
		# 			original_token = atk_row['orig_token']
		# 			pert_token = atk_row[f'{pert_key}_token']
		# 			blosum_score = get_blosum_score(original_token, pert_token)
		# 			print(blosum_score, end=" ")
		# 	print("\n\n")

		for key in perturbations_keys:
			assert len(atk_df[f'{key}_sequence'].unique())==1

		assert len(atk_df)==len(target_token_idxs)
		return atk_df, torch.tensor(embeddings_distances)

	def evaluate_missense(self, missense_row, msa, original_embedding, signed_gradient, adversarial_df, perturbations_keys, 
		p_norm=1, verbose=False):

		missense_row = missense_row.to_dict()   

		if verbose:
		  print("\n\n-- Evaluating missense mutation --")

		self.original_model.eval()
		self.embedding_model.eval()
		device = next(self.original_model.parameters()).device

		### compute embeddings

		original_sequence = adversarial_df['original_sequence'].unique()[0] 
		batch_converter = self.alphabet.get_batch_converter()

		with torch.no_grad():

			original_residue_idx = self.alphabet.get_idx(missense_row['original_token'])

			### mutation embedding

			batch = [("mutated_sequence", missense_row['mutated_sequence'])] + list(msa[1:])
			batch_labels, batch_strs, original_batch_tokens = batch_converter(batch)
			original_batch_tokens = original_batch_tokens.to(device)
			results = self.original_model(original_batch_tokens, repr_layers=[0], return_contacts=True)
			mutated_embedding = results["representations"][0].to(device)

			embedding_distance = torch.norm(mutated_embedding-original_embedding, p=p_norm).item()
			missense_row['original_embedding_distance'] = embedding_distance

			embeddings_diff = original_embedding-mutated_embedding
			cosine_similarity = nn.CosineSimilarity(dim=-1)(signed_gradient.flatten(), embeddings_diff.flatten()).item()
			missense_row['original_cosine_similarity'] = cosine_similarity

			blosum_distance = compute_blosum_distance(missense_row['mutated_sequence'], original_sequence,
				target_token_idxs=None) # [missense_row['mutation_idx']])
			missense_row['original_blosum_distance'] = blosum_distance

			if verbose:
				print(f"\nmutated vs original:\tembedding_distance = {embedding_distance}\tcosine_similarity = {cosine_similarity}\tblosum_distance = {blosum_distance}")

			### perturbations embeddings

			for pert_key in perturbations_keys:

				perturbed_sequence = adversarial_df[f'{pert_key}_sequence'].unique()[0] 

				assert len(missense_row['mutated_sequence']) == len(perturbed_sequence)

				batch = [(f"{pert_key}_sequence", perturbed_sequence)] + list(msa[1:])
				batch_labels, batch_strs, pert_batch_tokens = batch_converter(batch)
				pert_batch_tokens = pert_batch_tokens.to(device)
				results = self.original_model(pert_batch_tokens, repr_layers=[0], return_contacts=True)
				pert_embedding = results["representations"][0].to(device)

				embedding_distance = torch.norm(mutated_embedding-pert_embedding, p=p_norm).item()
				missense_row[f'{pert_key}_embedding_distance'] = embedding_distance

				embeddings_diff = pert_embedding-mutated_embedding
				cosine_similarity = nn.CosineSimilarity(dim=0)(signed_gradient.flatten(), embeddings_diff.flatten()).item()
				missense_row['original_cosine_similarity'] = cosine_similarity

				blosum_distance = compute_blosum_distance(missense_row['mutated_sequence'], perturbed_sequence, 
					target_token_idxs=None) # [missense_row['mutation_idx']])
				missense_row[f'{pert_key}_blosum_distance'] = blosum_distance

				if verbose:
					print(f"\nmutated vs {pert_key}:\tembedding_distance = {embedding_distance}\tcosine_similarity = {cosine_similarity}\tblosum_distance = {blosum_distance}")

		return missense_row