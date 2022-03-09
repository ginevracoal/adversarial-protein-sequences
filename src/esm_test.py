import os
import esm
import torch
import matplotlib.pyplot as plt
from embedding_model import EmbModel

esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

data = [
    ("protein", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = esm1_model(batch_tokens, repr_layers=[0], return_contacts=True)

print("\nresults.keys() =", results.keys())

# eval EmbModel on first embedding and check output logits are equal

token_representations = results["representations"][0]
print("\ntoken_representations.shape =", token_representations.shape)

first_embedding = token_representations
model = EmbModel(esm1_model, alphabet)
assert torch.all(torch.eq(model(first_embedding)['logits'], results['logits']))

# attack first embedding on token 5

epsilon = 0.1

first_embedding.requires_grad=True
output = model(first_embedding)

loss = torch.max(output['logits'])
print(loss)
model.zero_grad()
loss.backward()

perturbed_embedding = first_embedding + epsilon * first_embedding.grad.data.sign()

print(perturbed_embedding.shape)
print("\ndistance from the original embedding =", torch.norm(perturbed_embedding-first_embedding).item())

first_embedding.requires_grad=False

# find nearest hidden representation of perturbed sequences in the embedding space

chosen_token_idx = 5
original_sequence = data[0][1]

current_token = str(list(original_sequence)[chosen_token_idx])
print("\ncurrent_token =", current_token)
tokens_list = list(set(alphabet.standard_toks) - set(['.','-',current_token]))

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
# print("\ntoken_representations.shape =", token_representations.shape)

distances = torch.stack([torch.norm(perturbed_embedding-z_2) for z_2 in token_representations])
min_dist_idx = torch.argmin(distances)

print("\nnew token =", tokens_list[min_dist_idx])
print("\nadversarial sequence =", perturbed_data[min_dist_idx])
print("\ndistance between embeddings =", distances[min_dist_idx])






