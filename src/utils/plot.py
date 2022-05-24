import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_cosine_similarity(df, keys=['max_cos'], filepath=None, filename=None):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(5*len(keys), 4), ncols=len(keys), sharey=True)

    for i in range(len(keys)):
        axis = ax if len(keys)==1 else ax[i]
        df = df.sort_values(f'{keys[i]}_token') 
        sns.stripplot(data=df, x=f'{keys[i]}_token', y=f'{keys[i]}', dodge=True, ax=axis)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig

def plot_tokens_hist(df, keys, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    df['perc_token_idx'] = df.apply(lambda row: row['target_token_idx']/len(row['original_sequence']), axis=1)
    sns.histplot(data=df, x="perc_token_idx", legend=None)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(os.path.join(filepath, filename+".png"))
    plt.close()

    ### split by perturbations_keys
    assert len(keys)==4
    fig, ax = plt.subplots(figsize=(10, 7), nrows=2, ncols=2, sharey=True)
    jitter = 0.1

    for key, axis in ((keys[0],ax[0,0]), (keys[1],ax[1,0]), (keys[2],ax[0,1]), (keys[3],ax[1,1])):
        df = df.sort_values(f'{key}_token') 
        sns.stripplot(data=df, y="perc_token_idx", x=f"{key}_token", dodge=True, ax=axis, jitter=jitter)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+f"_token_split.png"))
        plt.close()

    return fig

def plot_token_substitutions(df, keys, filepath=None, filename=None):

    sns.set_style("darkgrid")

    for key in keys:

        fig, ax = plt.subplots(figsize=(10, 7))

        subst_counts = df.groupby(['orig_token', f'{key}_token'], as_index=False).size()
        subst_counts = subst_counts.rename(columns={"size": "n_substitutions"})

        df_heatmap = subst_counts.pivot_table(values='n_substitutions', columns='orig_token', index=f'{key}_token')
        sns.heatmap(df_heatmap, annot=True, cmap="rocket_r")

        fontdict = {'fontsize': 10}
        plt.tight_layout()
        plt.show()

        if filepath is not None and filename is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            fig.savefig(os.path.join(filepath, filename+f"_{key}_token.png"))
            plt.close()

def plot_cmap_distances(df, keys, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set(xlabel='k = len(sequence)-diag_idx', ylabel='l2 distance', 
        title='dist. bw original and perturbed contact maps')

    for key in keys:
        sns.lineplot(x=df['k'], y=df[f'{key}_cmap_dist'], label=key)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig

def plot_confidence(df, keys, filepath=None, filename=None):

    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, key in enumerate(keys):
        sns.distplot(x=df[f"{key}_pseudo_likelihood"], label=key, kde=True, hist=False)
    plt.xlabel('Pseudo likelihood of substitution')
    plt.tight_layout()
    plt.legend()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_pseudo_likelihood.png"))
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, key in enumerate(keys):
        sns.distplot(x=df[f"{key}_evo_velocity"], label=key, kde=True, hist=False)
    plt.xlabel('Pseudo likelihood of substitution')
    plt.tight_layout()
    plt.legend()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_evo_velocity.png"))
        plt.close()

def plot_embeddings_distances(embeddings_distances, filepath, filename):

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(x=embeddings_distances.flatten(), legend=None)
    plt.xlabel(r'Distribution of distances $||z-z_I(C_I)||_2$ (varying $z$ and $C_I$)')
    plt.tight_layout()
    plt.legend()
    plt.show()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(os.path.join(filepath, filename+".png"))
    plt.close()

def plot_blosum_distances(df, keys, filepath=None, filename=None, plot_method='histplot'):

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if plot_method=='histplot':
        df = df[['original_sequence']+[f"{key}_blosum_dist" for key in keys]]
        df = df.melt(id_vars=['original_sequence'], var_name="key", value_name="blosum")
        ax = sns.histplot(x=df["blosum"], hue=df["key"], kde=False, multiple="stack")

    elif plot_method=='distplot':
        for idx, key in enumerate(keys):
            sns.distplot(x=df[f"{key}_blosum_dist"], label=key, kde=True, hist=False)

    else:
        raise ValueError

    plt.xlabel('Blosum distance')
    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig

def plot_tokens_attention(sequence, attentions, layer_idx, filepath=None, filename=None):

    layer_attention_scores = attentions[:,layer_idx-1,:,:,:].squeeze().detach().cpu().numpy()

    fig = plt.figure(figsize=(20, 20))

    for idx, scores in enumerate(layer_attention_scores):
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

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig


# def plot_attention_matrix(attention_matrix, sequence, target_token_idxs, filepath=None, filename=None):

#     n_layers = attention_matrix.shape[0]

#     fig, ax = plt.subplots(figsize=(10,7))
#     ax = sns.heatmap(attention_matrix, linewidth=0.2)

#     ax.set_xticks(range(len(sequence)))
#     ax.set_yticks(range(n_layers))

#     fontdict = {'fontsize': 10}
#     ax.set_xticklabels(sequence, fontdict=fontdict)
#     ax.set_yticklabels(list(range(n_layers)), fontdict=fontdict, rotation=90)

#     for target_token_idx in target_token_idxs:
#         x, y, w, h = target_token_idx, 0, 1, n_layers
#         ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='black', lw=2, clip_on=False))
#         ax.tick_params(length=0)

#     plt.xlabel('Tokens')
#     plt.ylabel('Layers')

#     plt.tight_layout()
#     plt.show()  

#     if filepath is not None and filename is not None:
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#         fig.savefig(os.path.join(filepath, filename+".png"))
#         plt.close()

#     return fig

def plot_contact_maps(original_contacts, adversarial_contacts, filepath=None, filename=None):

    fig, ax = plt.subplots(figsize=(10, 4), ncols=3)
    ax[0].imshow(original_contacts, cmap="Blues")
    ax[1].imshow(adversarial_contacts, cmap="Blues")
    ax[2].imshow(original_contacts-adversarial_contacts, cmap="Blues")

    ax[0].set_xlabel('original')
    ax[1].set_xlabel('adversarial')
    ax[2].set_xlabel('orig-adv')

    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig