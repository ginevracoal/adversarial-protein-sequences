#!/usr/bin/python 

import os
import esm
import torch
import matplotlib.pyplot as plt

from utils.data import *
from utils.plot import *
from sequence_attack import SequenceAttack
from models.esm_embedding import EsmEmbedding
from utils.protein_sequences import compute_blosum_distance, get_contact_map

out_plots_path = '/fast/external/gcarbone/adversarial-protein-sequences_out/plots/toy_example/'

esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

batch_converter = alphabet.get_batch_converter()
n_layers = esm1_model.args.layers

data = [
    ("original_sequence", "TPEEFMLVYKFARKHHITLTNLITEETTHVVMKTDAEFVCERTLKYFLGIAGGKWVVSYFWVTQSI"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

original_sequence = data[0][1]

with torch.no_grad():
    results = esm1_model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=True)

print("\nresults.keys() =", results.keys())
print("\nattentions.shape = (batch x layers x heads x seq_len x seq_len) =", results['attentions'].shape)
print("\nrepresentations.shape = (batch, seq_len, hidden_size) =", results['representations'][0].shape)
print("\nlogits.shape = (batch, seq_len, n_tokens) =", results['logits'].shape)

first_embedding = results["representations"][0]

# instantiate model

model = EsmEmbedding(esm1_model, alphabet)
atk = SequenceAttack(original_model=esm1_model, embedding_model=model, alphabet=alphabet)

# target token idx

target_token_idxs, target_tokens_attention = atk.choose_target_token_idxs(batch_tokens=batch_tokens, 
    n_token_substitutions=3, target_attention='last_layer')
print("\ntarget_token_idxs =", target_token_idxs)

# attention scores

attentions = model.compute_attention_matrix(batch_tokens=batch_tokens, layers_idxs=[n_layers-1])
attentions = attentions.squeeze().cpu().detach().numpy()
plot_attention_grid(sequence=original_sequence, heads_attention=attentions, layer_idx=n_layers, 
    filepath=out_plots_path, target_token_idxs=target_token_idxs, filename=f"tokens_attention_layer={n_layers}")

# attack

signed_gradient, loss = atk.compute_loss_gradient(original_sequence=original_sequence, batch_tokens=batch_tokens, 
            target_token_idxs=target_token_idxs, first_embedding=first_embedding, loss_method='max_masked_prob')

perturbations_keys = ['masked_pred','max_cos','max_dist','max_cmap_dist'] 

atk_df, _ = atk.incremental_attack(name='seq', original_sequence=original_sequence, original_batch_tokens=batch_tokens,
    target_token_idxs=target_token_idxs, target_tokens_attention=target_tokens_attention,
    first_embedding=first_embedding, signed_gradient=signed_gradient, perturbations_keys=perturbations_keys, verbose=True)

print("\n\natk_df.keys():", atk_df.keys())

for key in perturbations_keys:
    adversarial_sequence = atk_df[f'{key}_sequence'].unique()[0]
    print(f"\n\n{key} sequence =", adversarial_sequence)

    # blosum distance

    blosum_distance = compute_blosum_distance(original_sequence, adversarial_sequence, target_token_idxs=target_token_idxs)
    print("\nblosum_distance = ", blosum_distance)

    # contact maps

    original_contacts = get_contact_map(model=esm1_model, alphabet=alphabet, sequence=original_sequence)
    adversarial_contacts = get_contact_map(model=esm1_model, alphabet=alphabet, sequence=adversarial_sequence)

    distance = torch.norm((original_contacts-adversarial_contacts).flatten(), p=1).item()
    print("\ndistance bw contact maps =", distance)

    fig = plot_cmaps(original_contacts=original_contacts.detach().numpy(), 
        adversarial_contacts=adversarial_contacts.detach().numpy(), 
        filepath=out_plots_path, filename=f"{key}", key=key)




