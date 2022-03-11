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

        print("\ntarget_token_idx =", target_token_idx)
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

        return signed_gradient

    def attack_sequence(self, original_sequence, target_token_idx, first_embedding, signed_gradient, 
        embedding_distance='cosine'):

        current_token = str(list(original_sequence)[target_token_idx])
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
            results = self.original_model(batch_tokens, repr_layers=[0], return_contacts=False)
            token_representations = results["representations"][0]

        if embedding_distance=='cosine':
            # maximize cosine similarity

            distances = []
            for z_c in token_representations:
                z_c_diff = first_embedding-z_c
                distances.append(nn.CosineSimilarity(dim=0)(signed_gradient.flatten(), z_c_diff.flatten()))
            distances = torch.stack(distances)
            char_idx = torch.argmax(distances)

        elif embedding_distance=='euclidean':
            # minimize euclidean distance

            epsilon = 0.1
            perturbed_embedding = first_embedding+epsilon*signed_gradient

            distances = torch.stack([torch.norm(perturbed_embedding-z_c) for z_c in token_representations])
            char_idx = torch.argmin(distances)

        else:
            raise NotImplementedError

        print(f"\nnew token at position {target_token_idx} =", tokens_list[char_idx])
        print("\nadversarial sequence =", perturbed_data[char_idx])
        print(f"\n{embedding_distance} distance between embeddings =", distances[char_idx].item())

        return perturbed_data[char_idx][1]

    def compute_contact_maps(self, original_sequence, adversarial_sequence):

        batch_converter = self.alphabet.get_batch_converter()

        batch_labels, batch_strs, batch_tokens = batch_converter([('orig',original_sequence)])
        original_contacts = self.original_model.predict_contacts(batch_tokens)[0]

        batch_labels, batch_strs, batch_tokens = batch_converter([('adv', adversarial_sequence)])
        adversarial_contacts = self.original_model.predict_contacts(batch_tokens)[0]

        distance = torch.norm((original_contacts-adversarial_contacts).flatten()).item()
        print("\nl2 distance bw contact maps =", distance)

        return original_contacts, adversarial_contacts
