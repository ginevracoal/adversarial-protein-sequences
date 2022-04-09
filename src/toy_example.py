import os
import esm
import torch
import matplotlib.pyplot as plt

from embedding_model import EmbModel
from sequence_attack import SequenceAttack
from plot_utils import plot_tokens_attention, plot_representations_norms, plot_contact_maps

plots_path='plots/'

esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
n_layers = esm1_model.args.layers

data = [
    ("original_protein", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
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

# istantiate models

model = EmbModel(esm1_model, alphabet)
model.check_correctness(original_model=esm1_model, batch_tokens=batch_tokens)

atk = SequenceAttack(original_model=esm1_model, embedding_model=model, alphabet=alphabet)

# attention scores

plot_tokens_attention(sequence=original_sequence, attentions=results['attentions'], layer_idx=n_layers, 
    filepath=plots_path, filename=f"tokens_attention_layer={n_layers}")

# target token idx

target_token_idxs, repr_norms_matrix = atk.choose_target_token_idxs(batch_tokens=batch_tokens, n_token_substitutions=3)
print("\ntarget_token_idxs =", target_token_idxs)

plot_representations_norms(norms_mat=repr_norms_matrix.cpu().detach().numpy(), sequence=original_sequence, 
    target_token_idxs=target_token_idxs, filepath=plots_path, filename="representations_norms")

# attack

signed_gradient, loss = atk.compute_loss_gradient(original_sequence=original_sequence, 
            target_token_idxs=target_token_idxs, first_embedding=first_embedding, loss='maxTokensRepr')

atk_df = atk.attack_sequence(name='seq', original_sequence=original_sequence, target_token_idxs=target_token_idxs, 
    first_embedding=first_embedding, signed_gradient=signed_gradient,  verbose=True)

adversarial_sequence = atk_df['original_sequence'].unique()[0]
print("\nadversarial sequence =", adversarial_sequence)

# blosum distance
blosum_distance = atk.compute_blosum_distance(original_sequence, adversarial_sequence)
print("\nblosum_distance = ", blosum_distance)

# contact maps

original_contacts = atk.compute_contact_maps(original_sequence)
adversarial_contacts = atk.compute_contact_maps(adversarial_sequence)

distance = torch.norm((original_contacts-adversarial_contacts).flatten()).item()
print("\nl2 distance bw contact maps =", distance)

fig = plot_contact_maps(original_contacts=original_contacts.detach().numpy(), 
    adversarial_contacts=adversarial_contacts.detach().numpy(), 
    filepath=plots_path, filename="contact_maps")




