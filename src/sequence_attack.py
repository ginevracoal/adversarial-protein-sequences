import torch
import itertools
import pandas as pd
import torch.nn as nn
from Bio.SubsMat import MatrixInfo


class SequenceAttack():

    def __init__(self, original_model, embedding_model, alphabet):
        self.original_model = original_model
        self.embedding_model = embedding_model
        self.alphabet = alphabet

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
            start, end = 4, 29
            output_representations = output['representations'][self.original_model.args.layers]
            output_representations = output_representations[0, 1:len(original_sequence)+1, start:end]
            loss = torch.sum(torch.abs(output_representations[target_token_idxs,:]))

        self.embedding_model.zero_grad()
        loss.backward()

        signed_gradient = first_embedding.grad.data.sign()
        first_embedding.requires_grad=False
        return signed_gradient, loss

    def attack_sequence(self, name, original_sequence, target_token_idxs, first_embedding, signed_gradient, verbose=False):

        batch_converter = self.alphabet.get_batch_converter()
        _, _, original_batch_tokens = batch_converter([("original", original_sequence)])

        start, end = 4, 29
        amino_acids_tokens = self.alphabet.all_toks[start:end]

        max_cos_similarity = -1
        min_euclidean_dist = 10e10
        max_euclidean_dist = 0

        pred_confidence, max_cos_confidence, min_dist_confidence, max_dist_confidence = 0, 0, 0, 0

        orig_tokens, pred_tokens, max_cos_tokens, min_dist_tokens, max_dist_tokens = [], [], [], [], []
        batch_tokens_masked = original_batch_tokens.clone()

        for target_token_idx in target_token_idxs:

            ### mask original sequence at target_token_idx

            orig_tokens.append(original_sequence[target_token_idx])
            batch_tokens_masked[0, target_token_idx] = self.alphabet.mask_idx

            ### allowed substitutions at target_token_idx 

            current_token = original_sequence[target_token_idx]
            allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))

            for i, new_token in enumerate(allowed_token_substitutions):
                original_sequence_list = list(original_sequence)
                original_sequence_list[target_token_idx] = new_token
                perturbed_sequence = "".join(original_sequence_list)

                with torch.no_grad():
                    batch_labels, batch_strs, batch_tokens = batch_converter([(f"{i}th-seq", perturbed_sequence)])

                    results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0])
                    z_c = results["representations"][0]

                    z_c_diff = first_embedding-z_c
                    cosine_similarity = nn.CosineSimilarity(dim=0)(signed_gradient.flatten(), z_c_diff.flatten())
                    euclidean_distance = torch.norm(z_c_diff, p=2)

                    ### substitutions that maximize cosine similarity w.r.t. gradient direction

                    if cosine_similarity > max_cos_similarity:
                        max_cos_similarity = cosine_similarity
                        max_cos_token = new_token
                        max_cos_sequence = perturbed_sequence

                    ### substitutions that minimize/maximize euclidean distance from the original embedding

                    if euclidean_distance < min_euclidean_dist:
                        min_euclidean_dist = euclidean_distance
                        min_dist_token = new_token
                        min_dist_sequence = perturbed_sequence

                    if euclidean_distance > max_euclidean_dist:
                        max_euclidean_dist = euclidean_distance
                        max_dist_token = new_token
                        max_dist_sequence = perturbed_sequence

            # updating one token at a time
            original_sequence = perturbed_sequence

            max_cos_tokens.append(max_cos_token)
            min_dist_tokens.append(min_dist_token)
            max_dist_tokens.append(max_dist_token)

        ### prediction on sequence masked at target_token_idxs

        masked_prediction = self.original_model(batch_tokens_masked.to(signed_gradient.device))
        predicted_sequence_list = list(original_sequence)

        for i, target_token_idx in enumerate(target_token_idxs):
            logits = results["logits"][0, 1:len(original_sequence)+1, start:end]
            probs = torch.softmax(logits, dim=-1)
            predicted_token_idx = probs[target_token_idx,:].argmax()
            predicted_token = self.alphabet.all_toks[start+predicted_token_idx]
            predicted_sequence_list[target_token_idx] = predicted_token
            pred_tokens.append(predicted_token)

            ### compute confidence scores

            logits = results["logits"][0, 1:len(original_sequence)+1, :]
            all_probs = torch.softmax(logits, dim=-1)
            pred_confidence += all_probs[target_token_idx, self.alphabet.get_idx(predicted_token)]
            max_cos_confidence += all_probs[target_token_idx, self.alphabet.get_idx(max_cos_tokens[i])]
            min_dist_confidence += all_probs[target_token_idx, self.alphabet.get_idx(min_dist_tokens[i])]
            max_dist_confidence += all_probs[target_token_idx, self.alphabet.get_idx(max_dist_tokens[i])]

        predicted_sequence = "".join(predicted_sequence_list)

        pred_confidence /= len(target_token_idxs)
        max_cos_confidence /= len(target_token_idxs)
        min_dist_confidence /= len(target_token_idxs)
        max_dist_confidence /= len(target_token_idxs)

        assert pred_confidence<=1
        assert max_cos_confidence<=1
        assert min_dist_confidence<=1
        assert max_dist_confidence<=1

        ### blosum distances

        pred_blosum = self.compute_blosum_distance(original_sequence, predicted_sequence)
        max_cos_blosum = self.compute_blosum_distance(original_sequence, max_cos_sequence)
        min_dist_blosum = self.compute_blosum_distance(original_sequence, min_dist_sequence)
        max_dist_blosum = self.compute_blosum_distance(original_sequence, max_dist_sequence)

        if verbose:
            print(f"\norig_tokens = {orig_tokens}")
            print(f"pred_tokens = {pred_tokens}\t\tlikelihood = {pred_confidence}\t\tblosum_dist = {pred_blosum}")
            print(f"max_cos_tokens = {max_cos_tokens}\tlikelihood = {max_cos_confidence}\tblosum_dist = {max_cos_blosum}")
            print(f"min_dist_tokens = {min_dist_tokens}\tlikelihood = {min_dist_confidence}\tblosum_dist = {min_dist_blosum}")
            print(f"max_dist_tokens = {max_dist_tokens}\tlikelihood = {max_dist_confidence}\tblosum_dist = {max_dist_blosum}")

        atk_df = pd.DataFrame()

        for i, token_idx in enumerate(target_token_idxs):

            row = {
                'name':name,
                'original_sequence':original_sequence,
                'target_token_idx':token_idx,
                'pred_token':pred_tokens[i],
                'pred_sequence':predicted_sequence,
                'pred_confidence':pred_confidence.item(),
                'pred_blosum':pred_blosum,
                'max_cos_token':max_cos_tokens[i], 
                'max_cos_sequence':max_cos_sequence, 
                'max_cos_similarity':max_cos_similarity.item(), 
                'max_cos_confidence':max_cos_confidence.item(),
                'max_cos_blosum':max_cos_blosum,
                'min_dist_token':min_dist_tokens[i],
                'min_dist_sequence':min_dist_sequence,
                'min_euclidean_dist':min_euclidean_dist.item(), 
                'min_dist_confidence':min_dist_confidence.item(),
                'min_dist_blosum':min_dist_blosum,
                'max_dist_token':max_dist_tokens[i], 
                'max_dist_sequence':max_dist_sequence,
                'max_euclidean_dist':max_euclidean_dist.item(),
                'max_dist_confidence':max_dist_confidence.item(),
                'max_dist_blosum':max_dist_blosum,
                }

            # if verbose:
            #     print("\n")
            #     [print(key, value) for key, value in row.items()]

            atk_df = atk_df.append(row, ignore_index=True)

        return atk_df

    def compute_contact_map(self, sequence):

        device = next(self.original_model.parameters()).device
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq',sequence)])
        contact_map = self.original_model.predict_contacts(batch_tokens.to(device))[0]
        return contact_map

    def compute_blosum_distance(self, seq1, seq2):

        def get_score(token1, token2):
            try:
                return blosum[(token1,token2)]
            except:
                return -100 # very unlikely substitution 

        blosum = MatrixInfo.blosum62
        assert len(seq1)==len(seq2)
        blosum_distance = sum([get_score(seq1[i],seq1[i])-get_score(seq1[i],seq2[i]) for i in range(len(seq1))])

        return blosum_distance