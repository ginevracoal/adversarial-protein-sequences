import os
import esm
import torch
import matplotlib.pyplot as plt
# from captum.attr import Saliency, configure_interpretable_embedding_layer

from embedding_model import EmbModel#, SaliencyWrapper
from plot_utils import plot_tokens_attention, plot_representations_norms


esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
n_layers = esm1_model.args.layers

data = [
    ("protein", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

original_sequence = data[0][1]

with torch.no_grad():
    results = esm1_model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=True)

print("\nresults.keys() =", results.keys())
print("\nattentions.shape = (batch x layers x heads x seq_len x seq_len) =", results['attentions'].shape)
print("\nrepresentations.shape = (batch, seq_len, hidden_size) =", results['representations'][0].shape)
print("\nlogits.shape = (batch, seq_len, n_layers) =", results['logits'].shape)

# eval EmbModel on first embedding and check output logits are equal

first_embedding = results["representations"][0]

with torch.no_grad():
    model = EmbModel(esm1_model, alphabet)

assert torch.all(torch.eq(model(first_embedding)['logits'], results['logits']))

# plot attention scores

last_layer_attention_scores = results['attentions'][:,n_layers-1,:,:,:].squeeze().detach().cpu().numpy()

path='plots/'
os.makedirs(os.path.dirname(path), exist_ok=True)

fig = plot_tokens_attention(scores_mat=last_layer_attention_scores, sequence=original_sequence)
fig.savefig(path+f"tokens_attention_layer={n_layers}.png")
plt.close()

# choose token that maximizes the norm of representations in the last layer

representations = torch.stack(list(results["representations"].values())).squeeze()
repr_norms = torch.norm(representations, dim=2)

fig = plot_representations_norms(norms_mat=repr_norms.cpu().detach().numpy(), sequence=original_sequence)
fig.savefig(path+f"representations_norms.png")
plt.close()

chosen_token_idx = repr_norms[32].argmax().item()

# attack first embedding on the chosen token

epsilon = 0.1

first_embedding.requires_grad=True
output = model(first_embedding)

loss = torch.max(output['logits'])
model.zero_grad()
loss.backward()

perturbed_embedding = first_embedding + epsilon * first_embedding.grad.data.sign()

print(perturbed_embedding.shape)
print("\ndistance from the original embedding =", torch.norm(perturbed_embedding-first_embedding).item())

first_embedding.requires_grad=False

# find nearest hidden representation of perturbed sequences in the embedding space

current_token = str(list(original_sequence)[chosen_token_idx])
print("\ncurrent_token =", current_token)
tokens_list = list(set(alphabet.standard_toks) - set(['.','-',current_token]))
print("\nother tokens:", tokens_list)

perturbed_data = []
for token in tokens_list:
    original_sequence_list = list(original_sequence)
    original_sequence_list[chosen_token_idx] = token
    new_sequence = "".join(original_sequence_list)
    perturbed_data.append((f"protein-{token}", new_sequence))

batch_labels, batch_strs, batch_tokens = batch_converter(perturbed_data)
with torch.no_grad():
    results = esm1_model(batch_tokens, repr_layers=[0], return_contacts=False)

token_representations = results["representations"][0]
distances = torch.stack([torch.norm(perturbed_embedding-z_2) for z_2 in token_representations])
min_dist_idx = torch.argmin(distances)

print("\nnew token =", tokens_list[min_dist_idx])
print("\nadversarial sequence =", perturbed_data[min_dist_idx])
print("\ndistance between embeddings =", distances[min_dist_idx].item())






