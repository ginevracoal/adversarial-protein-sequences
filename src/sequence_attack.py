import torch
import torch.nn as nn
from embedding_model import EmbModel


class SequenceAttack():

    def __init__(self, original_model, embedding_model, alphabet):
        self.original_model = original_model
        self.embedding_model = embedding_model
        self.alphabet = alphabet

    def choose_target_token_idx(self, batch_tokens):

        n_layers = self.original_model.args.layers

        with torch.no_grad():
            results = self.original_model(batch_tokens, repr_layers=list(range(n_layers)))

        representations = torch.stack(list(results["representations"].values())).squeeze()
        significant_tokens_repr = representations[:, 1:-1, :]

        # compute l2 norm of feature vectors
        repr_norms = torch.norm(significant_tokens_repr, dim=2, p=2) 

        # divide rows by max values (i.e. in each layer)
        repr_norms_matrix = torch.nn.functional.normalize(repr_norms, p=2, dim=1)
        
        # choose token idx that maximizes the sum of normalized scores in all layers
        target_token_idx = repr_norms_matrix.sum(dim=0).argmax().item()

        return target_token_idx, repr_norms_matrix

    def perturb_embedding(self, first_embedding):

        first_embedding.requires_grad=True
        output = self.embedding_model(first_embedding)

        loss = torch.max(output['logits'])
        self.embedding_model.zero_grad()
        loss.backward()

        signed_gradient = first_embedding.grad.data.sign()
        first_embedding.requires_grad=False

        # print(perturbed_embedding.shape)
        # print("\ndistance from the original embedding =", torch.norm(perturbed_embedding-first_embedding).item())

        return signed_gradient, loss

    def attack_sequence(self, original_sequence, target_token_idx, first_embedding, signed_gradient, verbose=False):

        current_token = str(list(original_sequence)[target_token_idx])

        if verbose:
            print(f"\ncurrent_token at position {target_token_idx} =", current_token)

        tokens_list = list(set(self.alphabet.standard_toks) - set(['.','-',current_token]))
        # print("\nother tokens:", tokens_list)

        perturbed_data = []
        for token in tokens_list:
            original_sequence_list = list(original_sequence)
            original_sequence_list[target_token_idx] = token
            new_sequence = "".join(original_sequence_list)
            perturbed_data.append((f"{token}-substitution", new_sequence))

        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_data)

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

            ### adv attacks maximize cosine similarity
            adv_char_idx = torch.argmax(cosine_distances)

            ### "safe" substitutions minimize cosine similarity
            min_cos_similarity_char_idx = torch.argmin(cosine_distances)

            ### substitutions s.t. the accociated fgsm attacks minimize euclidean distance from fgsm attacks with varying epsilon

            # min_eps, max_eps = torch.min(euclidean_distances).item(), torch.max(euclidean_distances).item()
            # epsilon_values = torch.range(min_eps, max_eps, step=(max_eps - min_eps) / 10)
            # print(epsilon_values)
            # min_euclidean_dist_char_idx = []
            # max_euclidean_dist_char_idx = []
            # for epsilon in epsilon_values:

            #     perturbed_embedding = first_embedding+epsilon*signed_gradient 
            #     euclidean_distances_from_attacks = [torch.norm(perturbed_embedding-z_c,p=2) for z_c in token_representations]
            #     euclidean_distances_from_attacks = torch.stack(euclidean_distances_from_attacks)
            #     min_euclidean_dist_char_idxs.append(torch.argmin(euclidean_distances_from_attacks))
            #     max_euclidean_dist_char_idxs.append(torch.argmax(euclidean_distances_from_attacks))

            # min_euclidean_dist_char_idx = torch.mode(torch.stack(min_euclidean_dist_char_idxs))[0]
            # max_euclidean_dist_char_idx = torch.mode(torch.stack(max_euclidean_dist_char_idxs))[0]

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

        print(atk_dict)

        if verbose:
            print(f"\nnew token at position {target_token_idx} = {adv_token}")
            
        return atk_dict

    def compute_contact_maps(self, sequence):

        device = next(self.original_model.parameters()).device

        # self.embedding_model.to('cpu')
        # self.original_model.to(device)
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([('seq',sequence)])
        contact_map = self.original_model.predict_contacts(batch_tokens.to(device))[0]

        return contact_map
