import torch
import itertools
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

    def perturb_embedding(self, first_embedding, verbose=False):

        first_embedding.requires_grad=True
        output = self.embedding_model(first_embedding)

        loss = torch.max(output['logits'])
        self.embedding_model.zero_grad()
        loss.backward()

        signed_gradient = first_embedding.grad.data.sign()
        first_embedding.requires_grad=False

        if verbose:
            print("\ndistance from the original embedding =", torch.norm(perturbed_embedding-first_embedding).item())

        return signed_gradient, loss

    def attack_sequence(self, original_sequence, target_token_idxs, first_embedding, signed_gradient, verbose=False):

        tokens_substitutions_dict = {}
        for target_token_idx in target_token_idxs:
            current_token = str(list(original_sequence)[target_token_idx])
            allowed_token_substitutions = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))

            tokens_substitutions_dict[str(target_token_idx)] = {'current_token':current_token, 
                'allowed_token_substitutions':allowed_token_substitutions}

        if verbose:
            print("\ntokens_substitutions_dict:\n", tokens_substitutions_dict)

        allowed_token_substitutions_list = [tokens_substitutions_dict[str(target_token_idx)]['allowed_token_substitutions'] \
            for target_token_idx in target_token_idxs]
        allowed_seq_substitutions = list(itertools.product(*allowed_token_substitutions_list))

        perturbed_sequences = []
        for i, sequence_substitution in enumerate(allowed_seq_substitutions):
            original_sequence_list = list(original_sequence)

            for target_token_idx, new_token in zip(target_token_idxs, sequence_substitution):
                original_sequence_list[target_token_idx] = new_token

            new_sequence = "".join(original_sequence_list)
            perturbed_sequences.append((f"{i}th-substitution", new_sequence))

        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_sequences)

        with torch.no_grad():
            results = self.original_model(batch_tokens.to(signed_gradient.device), repr_layers=[0], return_contacts=False)
            token_representations = results["representations"][0]

            euclidean_distances = []
            cosine_distances = []
            for z_c in token_representations:
                z_c_diff = first_embedding-z_c
                euclidean_distances.append(torch.norm(z_c_diff, p=2))
                cosine_distances.append(nn.CosineSimilarity(dim=0)(signed_gradient.flatten(), z_c_diff.flatten()))

            euclidean_distances = torch.stack(euclidean_distances)
            cosine_distances = torch.stack(cosine_distances)

            ### adv substitutions maximize cosine similarity w.r.t. gradient direction
            adv_char_idx = torch.argmax(cosine_distances)

            ### "safe" substitutions minimize cosine similarity w.r.t. gradient direction
            min_cos_similarity_char_idx = torch.argmin(cosine_distances)

            ### substitutions that minimize/maximize euclidean distance from classical fgsm attacks
            epsilon=torch.min(euclidean_distances).item()
            perturbed_embedding = first_embedding+epsilon*signed_gradient 
            euclidean_distances_from_attacks = [torch.norm(perturbed_embedding-z_c,p=2) for z_c in token_representations]
            euclidean_distances_from_attacks = torch.stack(euclidean_distances_from_attacks)
            min_euclidean_dist = torch.min(euclidean_distances_from_attacks)
            min_euclidean_dist_char_idx = torch.argmin(euclidean_distances_from_attacks)
            max_euclidean_dist = torch.max(euclidean_distances_from_attacks)
            max_euclidean_dist_char_idx = torch.argmax(euclidean_distances_from_attacks)

        atk_dict = {
            'adv_token':tokens_list[adv_char_idx], 
            'adv_sequence':perturbed_data[adv_char_idx][1], 
            'adv_cosine_distance':cosine_distances[adv_char_idx].item(), 
            'safe_token':tokens_list[min_cos_similarity_char_idx], 
            'safe_sequence':perturbed_data[min_cos_similarity_char_idx][1], 
            'safe_cosine_distance':cosine_distances[min_cos_similarity_char_idx].item(),
            'min_dist_token':tokens_list[min_euclidean_dist_char_idx], 
            'min_dist_sequence':perturbed_data[min_euclidean_dist_char_idx][1],
            'min_euclidean_dist':min_euclidean_dist.item(), 
            'max_dist_token':tokens_list[max_euclidean_dist_char_idx], 
            'max_dist_sequence':perturbed_data[max_euclidean_dist_char_idx][1],
            'max_euclidean_dist':max_euclidean_dist.item(), 
            }

        if verbose:
            print(f"\nnew token at position {target_token_idx} = {adv_token}")
            
        return atk_dict

    def compute_contact_maps(self, sequence):

        device = next(self.original_model.parameters()).device
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq',sequence)])
        contact_map = self.original_model.predict_contacts(batch_tokens.to(device))[0]
        return contact_map
