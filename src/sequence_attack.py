import torch
import itertools
import pandas as pd
import torch.nn as nn
from embedding_model import EmbModel


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

    def compute_embedding_gradient(self, first_embedding, verbose=False):

        first_embedding.requires_grad=True
        output = self.embedding_model(first_embedding)

        loss = torch.max(torch.abs(output['logits']))
        self.embedding_model.zero_grad()
        loss.backward()

        signed_gradient = first_embedding.grad.data.sign()
        first_embedding.requires_grad=False

        if verbose:
            print("\ndistance from the original embedding =", torch.norm(perturbed_embedding-first_embedding).item())

        return signed_gradient, loss

    def attack_sequence(self, original_sequence, target_token_idxs, first_embedding, signed_gradient, verbose=False):

        ### build dictionary of allowed tokens substitutions for each target token idx

        tokens_substitutions_dict = {}
        for target_token_idx in target_token_idxs:
            current_token = str(list(original_sequence)[target_token_idx])
            allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))

            tokens_substitutions_dict[f'token_{str(target_token_idx)}'] = {'current_token':current_token, 
                'allowed_token_substitutions':allowed_token_substitutions}

        allowed_token_substitutions_list = [tokens_substitutions_dict[f'token_{str(target_token_idx)}']['allowed_token_substitutions'] \
            for target_token_idx in target_token_idxs]
        allowed_seq_substitutions = list(itertools.product(*allowed_token_substitutions_list))

        ### build dict of all possible perturbed sequences

        perturbed_sequences_dict = {}
        for i, sequence_substitution in enumerate(allowed_seq_substitutions):
            original_sequence_list = list(original_sequence)

            for target_token_idx, new_token in zip(target_token_idxs, sequence_substitution):
                original_sequence_list[target_token_idx] = new_token

            new_sequence = "".join(original_sequence_list)
            perturbed_sequences_dict[f'seq_{str(i)}'] = {'new_tokens':sequence_substitution,'perturbed_sequence':new_sequence}

        if verbose:
            print("\ntokens_substitutions_dict:\n", tokens_substitutions_dict)
            print("\nn. perturbed_sequences =", len(perturbed_sequences_dict))
            # print("\nperturbed_sequences_dict first item:\n", list(perturbed_sequences_dict.items())[0])

        ### compute sequence attacks

        # min_eps, max_eps = 0.01, 0.1
        # epsilon_values = torch.arange(min_eps, max_eps, step=(max_eps - min_eps) / 10)
        # perturbed_embeddings = torch.stack([first_embedding+epsilon*signed_gradient for epsilon in epsilon_values])

        safe_cosine_similarity, adv_cosine_similarity = 1, -1
        min_euclidean_dist, max_euclidean_dist = 10e10, 0

        with torch.no_grad():
            for sequence_substitution in perturbed_sequences_dict.values():

                new_tokens = sequence_substitution['new_tokens']
                perturbed_sequence = sequence_substitution['perturbed_sequence']

                batch_converter = self.alphabet.get_batch_converter()
                batch_labels, batch_strs, batch_tokens = batch_converter([(f"{i}th-seq", perturbed_sequence)])

                results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0], return_contacts=False)
                z_c = results["representations"][0]

                z_c_diff = first_embedding-z_c
                cosine_similarity = nn.CosineSimilarity(dim=0)(signed_gradient.flatten(), z_c_diff.flatten())
                euclidean_distance = torch.norm(z_c_diff, p=2)
                # euclidean_distance = torch.mean(torch.stack([torch.norm(pert_emb-z_c,p=2) for pert_emb in perturbed_embeddings]))
                
                ### adv substitutions maximize cosine similarity w.r.t. gradient direction

                if cosine_similarity > adv_cosine_similarity:
                    adv_cosine_similarity = cosine_similarity
                    adv_tokens = new_tokens
                    adv_sequence = perturbed_sequence
                    # print("adv", new_tokens)

                ### "safe" substitutions minimize cosine similarity w.r.t. gradient direction

                if cosine_similarity < safe_cosine_similarity:
                    safe_cosine_similarity = cosine_similarity
                    safe_tokens = new_tokens
                    safe_sequence = perturbed_sequence
                    # print("safe", new_tokens)

                ### substitutions that minimize/maximize euclidean distance from classical fgsm attacks

                if euclidean_distance < min_euclidean_dist:
                    min_euclidean_dist = euclidean_distance
                    min_dist_tokens = new_tokens
                    min_dist_sequence = perturbed_sequence
                    # print("min", new_tokens)

                if euclidean_distance > max_euclidean_dist:
                    max_euclidean_dist = euclidean_distance
                    max_dist_tokens = new_tokens
                    max_dist_sequence = new_sequence
                    # print("max", new_tokens)

        if verbose:
            print("\nadv cosine_similarity =", adv_cosine_similarity, "\tadv_tokens =", adv_tokens)
            print("safe cosine_similarity =", safe_cosine_similarity, "\tsafe_tokens =", safe_tokens)
            print("min euclidean_distance =", min_euclidean_dist, "\tmin_dist_tokens =", min_dist_tokens)
            print("max euclidean_distance =", max_euclidean_dist, "\tmax_dist_tokens =", max_dist_tokens)

        atk_df = pd.DataFrame()

        for i, token_idx in enumerate(target_token_idxs):

            atk_df = atk_df.append({
                'original_sequence':original_sequence,
                'target_token_idx':token_idx,
                'adv_token':adv_tokens[i], 
                'adv_sequence':adv_sequence, 
                'adv_cosine_similarity':adv_cosine_similarity.item(), 
                'safe_token':safe_tokens[i],
                'safe_sequence':safe_sequence, 
                'safe_cosine_similarity':safe_cosine_similarity.item(),
                'min_dist_token':min_dist_tokens[i],
                'min_dist_sequence':min_dist_sequence,
                'min_euclidean_dist':min_euclidean_dist.item(), 
                'max_dist_token':max_dist_tokens[i], 
                'max_dist_sequence':max_dist_sequence,
                'max_euclidean_dist':max_euclidean_dist.item()
                }, ignore_index=True)

        return atk_df

    def compute_contact_map(self, sequence):

        device = next(self.original_model.parameters()).device
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq',sequence)])
        contact_map = self.original_model.predict_contacts(batch_tokens.to(device))[0]
        return contact_map
