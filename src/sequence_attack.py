import torch
import random
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from Bio.SubsMat import MatrixInfo

DEBUG=False

class SequenceAttack():

    def __init__(self, original_model, embedding_model, alphabet):
        original_model.eval()
        embedding_model.eval()

        self.original_model = original_model
        self.embedding_model = embedding_model
        self.alphabet = alphabet

        # start/end idxs of residues tokens in the alphabet
        self.start, self.end = 4, 29
        self.residues_tokens = self.alphabet.all_toks[self.start:self.end]

    def choose_target_token_idxs(self, batch_tokens, n_token_substitutions, target_attention, verbose=False):

        print("\nChoosing target token idxs.")

        n_layers = self.original_model.args.layers

        if target_attention=='all_layers':
            layers_idxs = list(range(n_layers))

        elif target_attention=='last_layer':
            layers_idxs = [n_layers-1]

        else:
            raise AttributeError("Wrong target attention.")

        with torch.no_grad():
            results = self.original_model(batch_tokens, repr_layers=layers_idxs, return_contacts=True)

        # compute tokens attention
        tokens_attention = self.embedding_model.get_tokens_attention(results=results, layers_idxs=layers_idxs, verbose=verbose)

        # choose top n_token_substitutions token idxs that maximize the sum of normalized scores
        target_token_idxs = torch.topk(tokens_attention, k=n_token_substitutions).indices.cpu().detach().numpy()

        if verbose:
            print(f"\ntarget_token_idxs = {target_token_idxs}")

        return target_token_idxs, tokens_attention

    def compute_loss_gradient(self, original_sequence, target_token_idxs, first_embedding, loss, verbose=False):

        first_embedding.requires_grad=True
        output = self.embedding_model(first_embedding=first_embedding, repr_layers=[self.original_model.args.layers])

        if loss=='maxLogits':
            loss = torch.max(torch.abs(output['logits']))

        elif loss=='maxTokensRepr':
            output_representations = output['representations'][self.original_model.args.layers].squeeze()
            output_representations = output_representations[1:len(original_sequence)+1, :]#self.start:self.end]
            loss = torch.sum(torch.abs(output_representations[target_token_idxs,:]))

        else:
            raise AttributeError

        self.embedding_model.zero_grad()
        loss.backward()

        signed_gradient = first_embedding.grad.data.sign()
        first_embedding.requires_grad=False
        return signed_gradient, loss

    def attack_sequence(self):
        raise NotImplementedError("Implement seq atk without evaluations")

    def attack_sequence(self, name, original_sequence, target_token_idxs, first_embedding, signed_gradient, 
        verbose=False, perturbations_keys=['pred', 'max_cos','min_dist','max_dist']):

        assert 'pred' in perturbations_keys

        adv_perturbations_keys = perturbations_keys.copy()
        adv_perturbations_keys.remove('pred')
        
        batch_converter = self.alphabet.get_batch_converter()
        _, _, original_batch_tokens = batch_converter([("original", original_sequence)])
        batch_tokens_masked = original_batch_tokens.clone().squeeze()
        assert len(batch_tokens_masked.shape)==1

        ### init dictionary
        atk_dict = {
            'name':name, 
            'original_sequence':original_sequence, 
            'orig_tokens':[], 
            'target_token_idxs':target_token_idxs}

        for pert_key in perturbations_keys:
            atk_dict.update({
                f'{pert_key}_tokens':[], 
                f'{pert_key}_sequence':original_sequence,
                f'{pert_key}_pseudo_likelihood':0., 
                f'{pert_key}_evo_velocity':0., 
                f'{pert_key}_blosum_dist':0.
                })

        embeddings_distances = []

        for target_token_idx in target_token_idxs:

            if DEBUG:
                print("\ntarget_token_idx =", target_token_idx)

            atk_dict['orig_tokens'].append(original_sequence[target_token_idx])

            ### mask original sequence at target_token_idx
            batch_tokens_masked[target_token_idx] = self.alphabet.mask_idx

            ### allowed substitutions at target_token_idx 
            current_token = original_sequence[target_token_idx]
            allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))

            for pert_key in adv_perturbations_keys:

                if DEBUG:
                    print("\tpert_key =", pert_key)

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
                        batch_labels, batch_strs, batch_tokens = batch_converter([(f"{i}th-seq", perturbed_sequence)])

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
                            new_token = token
                            
                            if DEBUG:
                                print("\t\tperturbed_sequence =", perturbed_sequence)

                        ### substitutions that minimize/maximize euclidean distance from the original embedding

                        if pert_key=='min_dist' and euclidean_distance < atk_dict[pert_key]:
                            atk_dict[pert_key] = euclidean_distance.item()
                            atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
                            new_token = token

                            if DEBUG:
                                print("\t\tperturbed_sequence =", perturbed_sequence)

                        if pert_key=='max_dist' and euclidean_distance > atk_dict[pert_key]:
                            atk_dict[pert_key] = euclidean_distance.item()
                            atk_dict[f'{pert_key}_sequence'] = perturbed_sequence
                            new_token = token

                            if DEBUG:
                                print("\t\tperturbed_sequence =", perturbed_sequence)
                                
                atk_dict[f'{pert_key}_tokens'].append(new_token)

        assert len(atk_dict[f'{pert_key}_tokens'])==len(target_token_idxs)
        
        ### prediction on sequence masked at target_token_idxs

        masked_prediction = self.original_model(batch_tokens_masked.to(signed_gradient.device))
        predicted_sequence_list = list(original_sequence)

        for i, target_token_idx in enumerate(target_token_idxs):
            logits = results["logits"][0, 1:len(original_sequence)+1, :]
            probs = torch.softmax(logits, dim=-1)

            predicted_token_idx = probs[target_token_idx,self.start:self.end].argmax()
            predicted_token = self.residues_tokens[predicted_token_idx]
            predicted_sequence_list[target_token_idx] = predicted_token
            atk_dict['pred_tokens'].append(predicted_token)

            ### compute confidence scores

            for pert_key in perturbations_keys:
                orig_token, new_token = original_sequence[target_token_idx], atk_dict[f'{pert_key}_tokens'][i]
                orig_residue_idx, new_residue_idx = self.alphabet.get_idx(orig_token), self.alphabet.get_idx(new_token)

                orig_log_prob = torch.log((probs[target_token_idx, orig_residue_idx]))
                adv_prob = (probs[target_token_idx, new_residue_idx])
                adv_log_prob = torch.log(adv_prob)

                atk_dict[f'{pert_key}_pseudo_likelihood'] += (adv_prob/len(target_token_idxs)).item()
                atk_dict[f'{pert_key}_evo_velocity'] += ((orig_log_prob-adv_log_prob)/len(target_token_idxs)).item()

                if atk_dict[f'orig_tokens']==atk_dict[f'{pert_key}_tokens']:
                    assert atk_dict[f'{pert_key}_evo_velocity']==0.

        ### compute blosum distances

        if verbose:
            print(f"\norig_tokens = {atk_dict['orig_tokens']}")

        for pert_key in perturbations_keys:
            atk_dict[f'{pert_key}_blosum_dist'] = self.compute_blosum_distance(
                original_sequence, atk_dict[f'{pert_key}_sequence'], target_token_idxs)

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

    def compute_blosum_distance(self, seq1, seq2, target_token_idxs, penalty=2):

        def get_score(token1, token2):
            try:
                return blosum[(token1,token2)]
            except:
                return penalty*min(blosum.values()) # very unlikely substitution 

        blosum = MatrixInfo.blosum62
        assert len(seq1)==len(seq2)
        blosum_distance = sum([get_score(seq1[i],seq1[i])-get_score(seq1[i],seq2[i]) for i in target_token_idxs])

        return blosum_distance

    def _compute_contact_map(self, sequence):

        device = next(self.original_model.parameters()).device
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq',sequence)])
        contact_map = self.original_model.predict_contacts(batch_tokens.to(device))[0]
        return contact_map

    def compute_cmaps_distance(self, atk_df, cmap_df, original_sequence, sequence_name, max_tokens, perturbations_keys,
        cmap_dist_lbound=0.2, cmap_dist_ubound=0.8):

        original_contact_map = self._compute_contact_map(sequence=original_sequence)
        cmap_dist_lbound = int(len(original_sequence)*cmap_dist_lbound)
        cmap_dist_ubound = int(len(original_sequence)*cmap_dist_ubound)
        min_k_idx, max_k_idx = len(original_sequence)-cmap_dist_ubound, len(original_sequence)-cmap_dist_lbound

        for k_idx, k in enumerate(np.arange(min_k_idx, max_k_idx, 1)):

            row_list = [['name', sequence_name],['k',k_idx]]
            for key in perturbations_keys:

                topk_original_contacts = torch.triu(original_contact_map, diagonal=k)
                new_contact_map = self._compute_contact_map(sequence=atk_df[f'{key}_sequence'].unique()[0])
                topk_new_contacts = torch.triu(new_contact_map, diagonal=k)
                cmap_distance = torch.norm((topk_original_contacts-topk_new_contacts).flatten()).item()
                row_list.append([f'{key}_cmap_dist', cmap_distance/k])

            cmap_df = cmap_df.append(dict(row_list), ignore_index=True)

        return cmap_df