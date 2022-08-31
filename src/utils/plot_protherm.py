import os
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DPI=150
TOP=0.92
FONT_SIZE=13
palette="mako"
sns.set_style("darkgrid")
sns.set_palette(palette, 5)
linestyles=['-', '--', '-.', ':', '-']
matplotlib.rc('font', **{'size': FONT_SIZE})


def plot_hist_position_ranks(df, keys, filepath=None, filename=None):

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    for idx, key in enumerate(keys):
        tmp_df = df[df['perturbation']==key]
        sns.histplot(x=tmp_df["protherm_idx_rank"], label=key)

    plt.tight_layout()
    plt.show()
    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_position_ranks.png"))
        plt.close()

    return fig


def plot_cmap_distances(df, keys, filepath=None, filename=None):

    df = df[df['k']>=5]
    df = df[df['k']<=20]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    ax.set(xlabel=r'Upper triangular matrix index $k$', #' = len(sequence)-diag_idx', 
        ylabel=r'dist(cmap($x$),cmap($\tilde{x}$))')

    for idx, key in enumerate(keys):
        tmp_df = df[df['perturbation']==key]
        sns.lineplot(x=tmp_df['k'], y=tmp_df[f'cmaps_distance'], label=key, ls=linestyles[idx])

    plt.tight_layout()
    plt.show()
    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_cmap_distances.png"))
        plt.close()

    return fig

def plot_confidence(df, keys, filepath=None, filename=None):

    ### perplexity

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    for idx, key in enumerate(keys):
        if key!='masked_pred':
            tmp_df = df[df['perturbation']==key]
            g = sns.distplot(x=tmp_df[f"perplexity"], label=key, kde=True, hist=False, 
                kde_kws={'linestyle':linestyles[idx]})
            # g.set(xlim=(0, None))

    plt.xlabel(r'Perplexity of predictions: $e^{H(p)}$')
    plt.tight_layout()
    plt.legend()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_perplexity.png"))
        plt.close()

    ### bleu

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    for idx, key in enumerate(keys):
        if key!='masked_pred':
            tmp_df = df[df['perturbation']==key]
            g = sns.distplot(x=tmp_df[f"bleu"], label=key, kde=True, hist=False, 
                kde_kws={'linestyle':linestyles[idx]})
            # g.set(xlim=(0, None))

    plt.xlabel(r'BLEU score')
    plt.tight_layout()
    plt.legend()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_bleu.png"))
        plt.close()

    ### pseudo-likelihood

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    for idx, key in enumerate(keys):
        hist=True if key=='masked_pred' else False
        tmp_df = df[df['perturbation']==key]
        sns.distplot(x=tmp_df[f"pseudo_likelihood"], label=key, kde=True, hist=hist, 
            kde_kws={'linestyle':linestyles[idx]})

    plt.xlabel(r'Pseudo likelihood of substitutions: $\mathbb{E}_{i\in I}[p(\tilde{x}_i|x^M_i)]$')
    plt.tight_layout()
    plt.legend()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_pseudo_likelihood.png"))
        plt.close()

    ### evo-velocity

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    for idx, key in enumerate(keys):
        hist=True if key=='masked_pred' else False
        tmp_df = df[df['perturbation']==key]
        sns.distplot(x=tmp_df[f"evo_velocity"], label=key, kde=True, hist=hist,
            kde_kws={'linestyle':linestyles[idx]})

    plt.xlabel(r'Evo velocity $\mathbb{E}_{i\in I}[ \log p(\tilde{x}_i|x^M_i)-\log p(x_i|x^M_i)]$')
    plt.tight_layout()
    plt.legend()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_evo_velocity.png"))
        plt.close()

def plot_embeddings_distances(df, keys, filepath, filename):
    matplotlib.rc('font', **{'size': FONT_SIZE})
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)

    ### adversarial perturbations
    for idx, key in enumerate(keys):
        tmp_df = df[df['perturbation']==key]
        hist=True if key=='masked_pred' else False      
        sns.distplot(x=tmp_df[f'embedding_distance'], label=key, kde=True, hist=hist,
            kde_kws={'linestyle':linestyles[idx]})

    ### all possible token choices and residues substitutions
    # sns.distplot(x=embeddings_distances.flatten(), label='perturb. embeddings', kde=True, hist=True)

    plt.xlabel(r'Embeddings distances: dist($z,\tilde{z}$)')
    plt.tight_layout()
    plt.legend()
    plt.show()
    # plt.yscale('log')

    # fig.suptitle(filename, fontsize=FONT_SIZE)
    # plt.subplots_adjust(top=TOP) 

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_embeddings_distances.png"))
        plt.close()

def plot_blosum_distances(df, keys, missense_df=None, filepath=None, filename=None, plot_method='distplot'):

    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    plt.xlabel(r'Blosum distance $(x,\tilde{x})$')

    for idx, key in enumerate(keys):
        hist=True if key=='masked_pred' else False
        tmp_df = df[df['perturbation']==key]
        sns.distplot(x=tmp_df["blosum_dist"], label=key, kde=True, hist=hist,
            kde_kws={'linestyle':linestyles[idx]})

    plt.legend()
    plt.tight_layout()
    plt.show()
    # fig.suptitle(filename, fontsize=FONT_SIZE)
    # plt.subplots_adjust(top=TOP) 

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_blosum_distances.png"))
        plt.close()

    return fig
