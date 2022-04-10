import torch
import itertools
import pandas as pd
import torch.nn as nn
from Bio.SubsMat import MatrixInfo

DEBUG=False

class SequenceAttack():

    def __init__(self, original_model, embedding_model, alphabet):
        self.original_model = original_model
        self.embedding_model = embedding_model
        self.alphabet = alphabet

        # start/end idxs of residues tokens in the alphabet
        self.start, self.end = 4, 29 
        self.residues_tokens = self.alphabet.all_toks[self.start:self.end]

    def choose_target_token_idxs(self, batch_tokens, n_token_substitutions, verbose=False):

        n_layers = self.original_model.args.layers

        with torch.no_grad():
            results = self.original_model(batch_tokens, repr_layers=list(range(n_layers)))

        representations = torch.stack(list(results["representations"].values())).squeeze()
        significant_tokens_repr = representations[:, 1:-1, :]

        # compute l2 norm of feature vectors
        repr_norms = torch.norm(significant_tokens_repr, dim=2, p=2) 

        # divide rows by max values (i.e. in each layer)
        repr_norms_matrix = torch.nn.functional.normalize(repr_norms, p=2, dim=1)
        
        # choose top n_token_substitutions token idxs that maximize the sum of normalized scores in all layers
        target_token_idxs = torch.topk(repr_norms_matrix.sum(dim=0), k=n_token_substitutions).indices.cpu().detach().numpy()

        if verbose:
            print(f"\ntarget_token_idxs = {target_token_idxs}")

        return target_token_idxs, repr_norms_matrix

    def compute_loss_gradient(self, original_sequence, target_token_idxs, first_embedding, loss, verbose=False):

        first_embedding.requires_grad=True
        output = self.embedding_model(first_embedding, repr_layers=[self.original_model.args.layers])

        if loss=='maxLogits':
            loss = torch.max(torch.abs(output['logits']))

        elif loss=='maxTokensRepr':
            output_representations = output['representations'][self.original_model.args.layers]
            output_representations = output_representations[0, 1:len(original_sequence)+1, self.start:self.end]
            loss = torch.sum(torch.abs(output_representations[target_token_idxs,:]))

        self.embedding_model.zero_grad()
        loss.backward()

        signed_gradient = first_embedding.grad.data.sign()
        first_embedding.requires_grad=False
        return signed_gradient, loss

    def attack_sequence(self, name, original_sequence, target_token_idxs, first_embedding, signed_gradient, verbose=False,
        perturbations_keys=['pred', 'max_cos','min_dist','max_dist'] ):

        assert 'pred' in perturbations_keys
        adv_perturbations_keys = perturbations_keys.copy()
        adv_perturbations_keys.remove('pred')

        batch_converter = self.alphabet.get_batch_converter()
        _, _, original_batch_tokens = batch_converter([("original", original_sequence)])
        batch_tokens_masked = original_batch_tokens.clone()

        atk_dict = {'name':name, 'original_sequence':original_sequence, 'orig_tokens':[], 'target_token_idxs':target_token_idxs}

        for pert_key in perturbations_keys:
            atk_dict.update({
                f'{pert_key}_tokens':[], 
                f'{pert_key}_sequence':original_sequence,
                f'{pert_key}_confidence':0, 
                f'{pert_key}_blosum':0
                })

            if pert_key=='max_cos':
                atk_dict.update({pert_key:-1})

            if pert_key=='min_dist':
                atk_dict.update({pert_key:10e10})

            if pert_key=='max_dist':
                atk_dict.update({pert_key:0})

        for target_token_idx in target_token_idxs:

            if DEBUG:
                print("\ntarget_token_idx =", target_token_idx)

            atk_dict['orig_tokens'].append(original_sequence[target_token_idx])

            ### mask original sequence at target_token_idx
            batch_tokens_masked[0, target_token_idx] = self.alphabet.mask_idx

            ### allowed substitutions at target_token_idx 
            current_token = original_sequence[target_token_idx]
            allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))

            for pert_key in adv_perturbations_keys:

                if DEBUG:
                    print("\tpert_key =", pert_key)

                # updating one token at a time
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
                new_residue_idx = self.alphabet.get_idx(atk_dict[f'{pert_key}_tokens'][i])
                atk_dict[f'{pert_key}_confidence'] += (probs[target_token_idx, new_residue_idx]/len(target_token_idxs)).item()

        atk_dict['pred_sequence'] = "".join(predicted_sequence_list)
        assert atk_dict[f'{pert_key}_confidence']<=1

        ### compute blosum distances

        if verbose:
            print(f"\norig_tokens = {atk_dict['orig_tokens']}")

        for pert_key in perturbations_keys:
            atk_dict[f'{pert_key}_blosum'] = self.compute_blosum_distance(
                original_sequence, atk_dict[f'{pert_key}_sequence'], target_token_idxs)

            if verbose:
                print(f"{pert_key}_tokens = {atk_dict[f'{pert_key}_tokens']}\
                    \tlikelihood = {atk_dict[f'{pert_key}_confidence']}\
                    \tblosum_dist = {atk_dict[f'{pert_key}_blosum']}")

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
        return atk_df

    def compute_contact_map(self, sequence):

        device = next(self.original_model.parameters()).device
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq',sequence)])
        contact_map = self.original_model.predict_contacts(batch_tokens.to(device))[0]
        return contact_map

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