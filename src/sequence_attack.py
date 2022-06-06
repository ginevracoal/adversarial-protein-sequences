import torch
import random
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn

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

	def choose_target_token_idxs(self, batch_tokens, n_token_substitutions, target_attention='last_layer', verbose=False):

		if verbose:
			print("\n=== Choosing target token idxs ===")

		n_layers = self.original_model.args.layers

		if target_attention=='all_layers':
			layers_idxs = list(range(n_layers))

		elif target_attention=='last_layer':
			layers_idxs = [n_layers-1]

		target_token_idxs, tokens_attention = self.embedding_model.get_target_token_idxs(batch_tokens=batch_tokens, 
			layers_idxs=layers_idxs, n_token_substitutions=n_token_substitutions, verbose=verbose)

		if verbose:
			print(f"\ntarget_token_idxs = {target_token_idxs}")

		return target_token_idxs, tokens_attention

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
		return signed_gradient, loss

	def attack_sequence(self, name, original_sequence, original_batch_tokens, target_token_idxs, first_embedding, 
		signed_gradient, msa=None, max_hamming_msa_size=None,
		verbose=False, perturbations_keys=['masked_pred', 'max_cos','min_dist','max_dist','max_entropy']):

		if verbose:
			print("\n=== Building adversarial sequences ===")

		assert 'masked_pred' in perturbations_keys
		adv_perturbations_keys = perturbations_keys.copy()
		adv_perturbations_keys.remove('masked_pred')
		
		if msa: 
			original_sequences=msa
			first_embedding=first_embedding[:,0]
		else:
			original_sequences=[("original", original_sequence)]

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

			atk_dict['orig_tokens'].append(original_sequence[target_token_idx])

			### mask original sequence at target_token_idx
			batch_tokens_masked = self.embedding_model.mask_batch_tokens(batch_tokens_masked, target_token_idx=target_token_idx)

			### allowed substitutions at target_token_idx 
			current_token = original_sequence[target_token_idx]
			allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))

			for pert_key in adv_perturbations_keys:

				if DEBUG:
					print("\n\tpert_key =", pert_key)
					print("\t\tcurrent_sequence =", atk_dict[f'{pert_key}_sequence'])

				if pert_key=='max_cos':
					atk_dict.update({pert_key:-1})

				if pert_key=='min_dist':
					atk_dict.update({pert_key:10e10})

				if pert_key=='max_dist':
					atk_dict.update({pert_key:0})

				### updating one token at a time
				for i, token in enumerate(allowed_token_substitutions):
					current_sequence_list = list(atk_dict[f'{pert_key}_sequence'])
					current_sequence_list[target_token_idx] = token
					perturbed_sequence = "".join(current_sequence_list)

					with torch.no_grad():

						if msa:
							perturbed_batch = get_max_hamming_msa(reference_sequence=(f"{i}th-seq", perturbed_sequence), 
								msa=msa, max_size=max_hamming_msa_size)
							batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)
							results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
							z_c = results["representations"][0][:,0]

						else:
							perturbed_batch = [(f"{i}th-seq", perturbed_sequence)]
							batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_batch)
							results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
							z_c = results["representations"][0]

						z_c_diff = first_embedding-z_c
						cosine_similarity = nn.CosineSimilarity(dim=0)(signed_gradient.flatten(), z_c_diff.flatten())
						euclidean_distance = torch.norm(z_c_diff, p=2)
						embeddings_distances.append(euclidean_distance)

						### substitutions that maximize cosine similarity w.r.t. gradient direction

						if pert_key=='max_cos' and cosine_similarity > atk_dict[pert_key]:
							atk_dict[pert_key] = cosine_similarity.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = euclidean_distance.item()
							new_token = token
							
							if DEBUG:
								print("\t\tperturbed_sequence =", perturbed_sequence)

						### substitutions that minimize/maximize euclidean distance from the original embedding

						if pert_key=='min_dist' and euclidean_distance < atk_dict[pert_key]:
							atk_dict[pert_key] = euclidean_distance.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = euclidean_distance.item()
							new_token = token

							if DEBUG:
								print("\t\tperturbed_sequence =", perturbed_sequence)

						if pert_key=='max_dist' and euclidean_distance > atk_dict[pert_key]:
							atk_dict[pert_key] = euclidean_distance.item()
							atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
							atk_dict[f'{pert_key}_embedding_distance'] = euclidean_distance.item()
							new_token = token

							if DEBUG:
								print("\t\tperturbed_sequence =", perturbed_sequence)
								
				atk_dict[f'{pert_key}_tokens'].append(new_token)

		assert len(atk_dict[f'{pert_key}_tokens'])==len(target_token_idxs)
		
		### prediction on sequence masked at target_token_idxs

		if DEBUG:
			print("\nbatch_tokens_masked =", batch_tokens_masked)
			print("\nbatch_chars_masked = ", [self.alphabet.all_toks[idx] for idx in batch_tokens_masked[0]])

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
				print(f"pred={predicted_token}, true={original_sequence[target_token_idx]}, acc={atk_dict['masked_pred_accuracy']}")

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
				atk_dict[f'{pert_key}_evo_velocity'] += ((orig_log_prob-adv_log_prob)/len(target_token_idxs)).item()

				if atk_dict[f'orig_tokens']==atk_dict[f'{pert_key}_tokens']:
					assert atk_dict[f'{pert_key}_evo_velocity']==0.

		predicted_sequence = "".join(predicted_sequence_list)

		if msa:
			predicted_batch = get_max_hamming_msa(reference_sequence=("pred_seq", predicted_sequence), msa=msa, 
				max_size=max_hamming_msa_size)
		else:
			predicted_batch = [(f"pred_seq", predicted_sequence)]
		
		batch_labels, batch_strs, batch_tokens = batch_converter(predicted_batch)
		results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
		z_c = results["representations"][0]
		euclidean_distance = torch.norm(first_embedding-z_c, p=2)

		atk_dict[f'masked_pred_sequence'] = predicted_sequence
		atk_dict[f'masked_pred_embedding_distance'] = euclidean_distance.item()

		### compute blosum distances

		if verbose:
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
