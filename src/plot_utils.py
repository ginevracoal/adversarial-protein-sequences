import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_tokens_attention(scores_mat, sequence):
    fig = plt.figure(figsize=(20, 20))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(5, 4, idx+1)
        im = ax.imshow(scores, cmap='viridis')

        ax.set_xticks(range(len(sequence)))
        ax.set_yticks(range(len(sequence)))

        fontdict = {'fontsize': 10}
        ax.set_xticklabels(sequence, fontdict=fontdict)
        ax.set_yticklabels(sequence, fontdict=fontdict, rotation=90)
        ax.set_xlabel('{} {}'.format('Head', idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    return fig


def plot_representations_norms(norms_mat, sequence, chosen_idx):
    n_layers = norms_mat.shape[0]

    fig, ax = plt.subplots(figsize=(10,7))
    ax = sns.heatmap(norms_mat, linewidth=0.2)

    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(n_layers))

    fontdict = {'fontsize': 10}
    ax.set_xticklabels(sequence, fontdict=fontdict)
    ax.set_yticklabels(list(range(n_layers)), fontdict=fontdict, rotation=90)

    x, y, w, h = chosen_idx, 0, 1, n_layers
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='black', lw=2, clip_on=False))
    ax.tick_params(length=0)

    plt.xlabel('Tokens')
    plt.ylabel('Layers')

    plt.tight_layout()
    plt.show()  

    return fig

def plot_contact_maps(original_contacts, adversarial_contacts):

    fig, ax = plt.subplots(figsize=(10, 6), ncols=2)
    ax[0].imshow(original_contacts, cmap="Blues")
    ax[1].imshow(adversarial_contacts, cmap="Blues")
    plt.show()
    return fig