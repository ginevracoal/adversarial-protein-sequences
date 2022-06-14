import torch
import random
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from Bio.SubsMat import MatrixInfo

from utils.protein import compute_blosum_distance, compute_cmaps_distance, get_max_hamming_msa

DEBUG=False


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
			print("\n=== Choosing target token idxs ===")

		layers_idxs = self._get_attention_layers_idxs(target_attention)
		tokens_attention = self.embedding_model.compute_tokens_attention(batch_tokens=batch_tokens, layers_idxs=layers_idxs)

		if token_selection=='max_attention':

			target_tokens_attention, target_token_idxs = torch.topk(tokens_attention, k=n_token_substitutions, largest=True)

		elif token_selection=='min_entropy':

			if msa is None:
				raise AttributeError("Entropy selection needs an MSA")

			tokens_entropy = self.compute_tokens_entropy(msa=msa, n_token_substitutions=n_token_substitutions)
			target_tokens_entropy, target_token_idxs = torch.topk(tokens_entropy, k=n_token_substitutions, largest=False)
			target_tokens_attention = tokens_attention[target_token_idxs]

		elif token_selection=='max_entropy':

			if msa is None:
				raise AttributeError("Entropy selection needs an MSA")

			tokens_entropy = self.compute_tokens_entropy(msa=msa, n_token_substitutions=n_token_substitutions)
			target_tokens_entropy, target_token_idxs = torch.topk(tokens_entropy, k=n_token_substitutions, largest=True)
			target_tokens_attention = tokens_attention[target_token_idxs]

		else:
			raise AttributeError("Wrong token_selection method")

		if verbose:
			print(f"\ntarget_token_idxs = {target_token_idxs}")

		return target_token_idxs.cpu().detach().numpy(), target_tokens_attention.cpu().detach().numpy()

	def compute_tokens_entropy(self, msa, n_token_substitutions):

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

	def compute_loss_gradient(self, original_sequence, target_token_idxs, first_embedding, loss_method, verbose=False):

		if verbose:
			print("\n=== Computing loss gradients ===")

		first_embedding.requires_grad=True
		output = self.embedding_model(first_embedding=first_embedding, repr_layers=[self.original_model.args.layers])
		loss = self.embedding_model.loss(method=loss_method, output=output, target_token_idxs=target_token_idxs)

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

		allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))
		
		if blosum_check:
			blosum = MatrixInfo.blosum62

			for new_token in allowed_token_substitutions:
				if ((current_token, new_token) not in blosum.keys()) & ((new_token, current_token) not in blosum.keys()):
					allowed_token_substitutions = list(set(allowed_token_substitutions) - set([new_token]))

		if len(allowed_token_substitutions)==0:
			print("\nNo substitutions allowed, keeping the original token fixed.")
			allowed_token_substitutions = [current_token]

		if DEBUG:
			print("\nallowed_token_substitutions =", allowed_token_substitutions)

		return allowed_token_substitutions

	def attack_sequence(self, name, original_sequence, original_batch_tokens, target_token_idxs, target_tokens_attention,
		first_embedding, signed_gradient, msa=None, verbose=False, 
		perturbations_keys=['masked_pred','max_cos','min_dist','max_dist'], p_norm=1):

		self.original_model.eval()
		self.embedding_model.eval()

		if verbose:
			print("\n=== Building adversarial sequences ===")

		adv_perturbations_keys = perturbations_keys.copy()

		assert 'masked_pred' in perturbations_keys
		adv_perturbations_keys.remove('masked_pred')
		
		if msa: 
			first_embedding=first_embedding[:,0]
			signed_gradient=signed_gradient[:,0]

		batch_converter = self.embedding_model.alphabet.get_batch_converter()
		batch_tokens_masked = original_batch_tokens.clone()

		### init dictionary
		atk_dict = {
			'name':name,
			'original_sequence':original_sequence,
			'orig_tokens':[],
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

		embeddings_distances = []

		for target_token_idx in target_token_idxs:

			current_token = original_sequence[target_token_idx]
			atk_dict['orig_tokens'].append(current_token)

			### mask original sequence at target_token_idx
			batch_tokens_masked = self.embedding_model.mask_batch_tokens(batch_tokens_masked, 
				target_token_idx=target_token_idx)

			### allowed substitutions at target_token_idx 
			allowed_token_substitutions = self.get_allowed_token_substitutions(current_token)

			for pert_key in adv_perturbations_keys:

				if DEBUG:
					print("\n\tpert_key =", pert_key)
					print(f"\t\tcurrent token at position {target_token_idx} = {current_token}")

				if pert_key=='max_cos':
					atk_dict.update({pert_key:-1})

				if pert_key=='min_dist':
					atk_dict.update({pert_key:10e10})

				if pert_key=='max_dist':
					atk_dict.update({pert_key:0})

				### updating one token at a time
				for j, token in enumerate(allowed_token_substitutions):
					current_sequence_list = list(atk_dict[f'{pert_key}_sequence'])
					current_sequence_list[target_token_idx] = token
					perturbed_sequence = "".join(current_sequence_list)

					with torch.no_grad():

						if msa:
							perturbed_batch = [("pert_seq", perturbed_sequence)] + list(msa[1:])
							batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)
							results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
							z_c = results["representations"][0][:,0]

						else:
							perturbed_batch = [("pert_seq", perturbed_sequence)]
							batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)
							results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
							z_c = results["representations"][0]

						z_c_diff = first_embedding-z_c
						cosine_similarity = nn.CosineSimilarity(dim=0)(signed_gradient.flatten(), z_c_diff.flatten())
						embedding_distance = torch.norm(z_c_diff, p=p_norm)
						embeddings_distances.append(embedding_distance)

						### substitutions that maximize cosine similarity w.r.t. gradient direction

						if pert_key=='max_cos' and cosine_similarity > atk_dict[pert_key]:
							atk_dict[pert_key] = cosine_similarity.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token
							
							if DEBUG:
								print(f"\t\tnew token at position {target_token_idx} = {new_token}\tcos_similarity = {cosine_similarity}")

						### substitutions that minimize/maximize distance from the original embedding

						if pert_key=='min_dist' and embedding_distance < atk_dict[pert_key]:
							atk_dict[pert_key] = embedding_distance.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token

							if DEBUG:
								print(f"\t\tnew token at position {target_token_idx} = {new_token}\tdistance = {embedding_distance}")

						if pert_key=='max_dist' and embedding_distance > atk_dict[pert_key]:
							atk_dict[pert_key] = embedding_distance.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = embedding_distance.item()
							new_token = token

							if DEBUG:
								print(f"\t\tnew token at position {target_token_idx} = {new_token}\tdistance = {embedding_distance}")

				atk_dict[f'{pert_key}_tokens'].append(new_token)

		assert len(atk_dict[f'{pert_key}_tokens'])==len(target_token_idxs)
		
		### prediction on sequence masked at target_token_idxs

		masked_prediction = self.original_model(batch_tokens_masked.to(signed_gradient.device))
		predicted_sequence_list = list(original_sequence)

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
				print("\n\tpert_key = masked_pred")
				print(f"\t\tpred_token = {predicted_token}, true_token = {original_sequence[target_token_idx]}, masked_pred_acc = {atk_dict['masked_pred_accuracy']}")

			### compute confidence scores

			for pert_key in perturbations_keys:

				orig_token = original_sequence[target_token_idx]
				new_token = atk_dict[f'{pert_key}_tokens'][i]

				orig_residue_idx = self.alphabet.get_idx(orig_token)
				new_residue_idx = self.alphabet.get_idx(new_token)

				orig_log_prob = torch.log(probs[target_token_idx, orig_residue_idx])
				adv_prob = probs[target_token_idx, new_residue_idx]
				adv_log_prob = torch.log(adv_prob)

				atk_dict[f'{pert_key}_pseudo_likelihood'] += (adv_prob/len(target_token_idxs)).item()
				atk_dict[f'{pert_key}_evo_velocity'] += ((adv_log_prob-orig_log_prob)/len(target_token_idxs)).item()

				if atk_dict[f'orig_tokens']==atk_dict[f'{pert_key}_tokens']:
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
			print(f"\norig_tokens = {atk_dict['orig_tokens']}\tmasked_pred_accuracy = {atk_dict['masked_pred_accuracy']}")

		for pert_key in perturbations_keys:
			atk_dict[f'{pert_key}_blosum_dist'] = compute_blosum_distance(original_sequence, 
				atk_dict[f'{pert_key}_sequence'], target_token_idxs)

			if verbose:
				print(f"\n{pert_key}\t", end="\t")
				for dict_key in ['tokens','pseudo_likelihood','evo_velocity','blosum_dist']:
					print(f"{dict_key} = {atk_dict[f'{pert_key}_{dict_key}']}", end="\t")

		### unstack tokens lists

		atk_df = pd.DataFrame()
		for i, token_idx in enumerate(target_token_idxs):
			row = atk_dict.copy()

			token_idx = row['target_token_idxs'][i]
			row['target_token_idx'] = row.pop('target_token_idxs')
			row['target_token_idx'] = token_idx
			row['target_token_attention'] = target_tokens_attention[i]

			token = row['orig_tokens'][i]
			row['orig_token'] = row.pop('orig_tokens')
			row['orig_token'] = token

			for pert_key in perturbations_keys:
				token = row[f'{pert_key}_tokens'][i]
				row[f'{pert_key}_token'] = row.pop(f'{pert_key}_tokens')
				row[f'{pert_key}_token'] = token

			atk_df = atk_df.append(row, ignore_index=True)

		assert len(atk_df)==len(target_token_idxs)
		return atk_df, torch.tensor(embeddings_distances)

	def evaluate_missense_mutation(self, name, original_sequence, mutated_sequence, target_token_idx,
		original_token, mutated_token, msa, target_attention, p_norm=1, verbose=False):

		self.original_model.eval()
		self.embedding_model.eval()
		device = next(self.original_model.parameters()).device

		### original prediction

		batch_converter = self.alphabet.get_batch_converter()

		with torch.no_grad():
			batch_labels, batch_strs, original_batch_tokens = batch_converter(msa)
			original_batch_tokens = original_batch_tokens.to(device)
			results = self.original_model(original_batch_tokens, repr_layers=[0], return_contacts=True)
			original_embedding = results["representations"][0].to(device)

		### attention

		layers_idxs = self._get_attention_layers_idxs(target_attention)
		tokens_attention = self.embedding_model.compute_tokens_attention(original_batch_tokens, layers_idxs=layers_idxs)
		token_attention = tokens_attention[target_token_idx]
		
		### masked prediction

		batch_tokens_masked = self.embedding_model.mask_batch_tokens(batch_tokens=original_batch_tokens.clone(), 
			target_token_idx=target_token_idx)

		with torch.no_grad():
			masked_prediction = self.original_model(batch_tokens_masked.to(device))

		logits = masked_prediction["logits"][:,0].squeeze() if msa else masked_prediction["logits"].squeeze()
		assert len(logits.shape)==2
		logits = logits[1:len(original_sequence)+1, :]
		probs = torch.softmax(logits, dim=-1)

		### pseudo likelihood + evo velocity

		original_residue_idx = self.alphabet.get_idx(original_token)
		mutated_residue_idx = self.alphabet.get_idx(mutated_token)
		original_prob = probs[target_token_idx, original_residue_idx]
		missense_prob = probs[target_token_idx, mutated_residue_idx]

		pseudo_likelihood = missense_prob
		evo_velocity = (torch.log(missense_prob)-torch.log(original_prob))

		### embedding + blosum distance

		if msa:
			assert len(mutated_sequence)==len(msa[1][1])
			missense_batch = [("missense_seq", mutated_sequence)] + list(msa[1:])
			batch_labels, batch_strs, missense_batch_tokens = batch_converter(missense_batch)
			results = self.original_model(missense_batch_tokens.to(device), repr_layers=[0])
			z_c = results["representations"][0][:,0]

		else:
			missense_batch = [("missense_seq", mutated_sequence)]
			batch_labels, batch_strs, missense_batch_tokens = batch_converter(missense_batch)
			results = self.original_model(missense_batch_tokens.to(device), repr_layers=[0])
			z_c = results["representations"][0]

		embedding_distance = torch.norm(original_embedding-z_c, p=p_norm)

		blosum_distance = compute_blosum_distance(original_sequence, mutated_sequence, target_token_idxs=[target_token_idx])

		evaluation_dict = {
			'target_token_attention':token_attention.item(), 
			'missense_pseudo_likelihood':pseudo_likelihood.item(),
			'missense_evo_velocity':evo_velocity.item(),
			'missense_embedding_distance':embedding_distance.item(),
			'missense_blosum_distance':blosum_distance,
			}

		if verbose:
			for key, value in evaluation_dict.items():
				print(f"{key} = {value}", end="\t")
			print("\n")

		return evaluation_dict