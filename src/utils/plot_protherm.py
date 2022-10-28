import os
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.plot import set_boxplot_linecolor

DPI=150
TOP=0.92
FONT_SIZE=13
palette="rocket_r"
sns.set_style("darkgrid")
sns.set_palette(palette, 5)
linestyles=['-', '--', '-.', ':', '-', '--']
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

def plot_cmap_distances(df, keys, distances_df_protherm=None, distances_df_adversarial=None, filepath=None, filename=None, 
	L=1, R=50, ci=None):

	df = df[(df['k']>=L) & (df['k']<=R)]

	fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
	ax.set(xlabel=r'Upper triangular matrix index $k$', ylabel=r'dist(cmap($x$),cmap($\tilde{x}$))')

	for idx, key in enumerate(keys):
		tmp_df = df[df['perturbation']==key]
		sns.lineplot(x=tmp_df['k'], y=tmp_df[f'cmaps_distance'], label=key, ls=linestyles[idx], ci=ci)

	if distances_df_adversarial is not None:
		distances_df_adversarial = distances_df_adversarial[(distances_df_adversarial['k']>=l_ths) & (distances_df_adversarial['k']<=r_ths)]
		sns.lineplot(x=distances_df_adversarial['k'], y=distances_df_adversarial[f'cmaps_distance'], label='other', 
			ls=linestyles[idx+1], ci=ci)
	
	if distances_df_protherm is not None:
		distances_df_protherm = distances_df_protherm[(distances_df_protherm['k']>=l_ths) & (distances_df_protherm['k']<=r_ths)]
		sns.lineplot(x=distances_df_protherm['k'], y=distances_df_protherm[f'cmaps_distance'], label='protherm', 
			ls=linestyles[idx+2], ci=ci)
	
	plt.tight_layout()
	plt.show()
	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_cmap_distances.png"))
		plt.close()

	return fig

def plot_confidence(df, keys, filepath=None, filename=None, plot_method='boxplot'):

	### perplexity

	if plot_method=='boxplot':

		fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
		sns.boxplot(data=df, y='perplexity', x="perturbation", palette=palette)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
		for idx, key in enumerate(keys):
			if key!='masked_pred':
				tmp_df = df[df['perturbation']==key]
				g = sns.distplot(x=tmp_df[f"perplexity"], label=key, kde=True, hist=False, 
					kde_kws={'linestyle':linestyles[idx]})

		plt.legend()
	
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_perplexity.png"))
		plt.close()

	### pseudo-likelihood

	if plot_method=='boxplot':

		fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
		sns.boxplot(data=df, y='pseudo_likelihood', x="perturbation", palette=palette)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=20)

		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
		for idx, key in enumerate(keys):
			hist=True if key=='masked_pred' else False
			tmp_df = df[df['perturbation']==key]
			sns.distplot(x=tmp_df[f"pseudo_likelihood"], label=key, kde=True, hist=hist, 
				kde_kws={'linestyle':linestyles[idx]})

		plt.xlabel(r'Pseudo likelihood of substitutions: $\mathbb{E}_{i\in I}[p(\tilde{x}_i|x^M_i)]$')
		plt.legend()

	plt.tight_layout()
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

def plot_embeddings_distances(df, keys, filepath, filename, distances_df_protherm=None, distances_df_adversarial=None, 
	plot_method='distplot'):

	if 'masked_pred' in keys:
		keys.remove('masked_pred')

	if plot_method=='boxplot':

		fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
		new_df = df[["perturbation", "embedding_distance"]]

		if distances_df_adversarial is not None:
			entry = pd.DataFrame({"embedding_distance":distances_df_adversarial['embedding_distance']}).assign(perturbation='other') 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		if distances_df_protherm is not None:
			entry = pd.DataFrame({"embedding_distance":distances_df_protherm['embedding_distance']}).assign(perturbation='protherm') 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		print(new_df)
		sns.boxplot(data=new_df, y='embedding_distance', x="perturbation", palette=palette)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 3.5), dpi=DPI)

		### adversarial perturbations
		for idx, key in enumerate(keys):
			tmp_df = df[df['perturbation']==key]
			hist=False #True if key=='protherm' else False      
			sns.distplot(x=tmp_df[f'embedding_distance'], label=key, kde=True, hist=hist,
				kde_kws={'linestyle':linestyles[idx]})

		if distances_df_adversarial is not None:
			sns.distplot(x=distances_df_adversarial['embedding_distance'], label='other', kde=True, hist=hist,
				kde_kws={'linestyle':linestyles[idx+1]})

		if distances_df_protherm is not None:
			sns.distplot(x=distances_df_protherm['embedding_distance'], label='protherm', kde=True, hist=hist,
				kde_kws={'linestyle':linestyles[idx+2]})

		plt.xlabel(r'Embeddings distance')
		plt.legend()
	
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_embeddings_distances.png"))
		plt.close()

def plot_blosum_distances(df, keys,  distances_df_protherm=None, distances_df_adversarial=None, filepath=None, filename=None, 
	plot_method='distplot'):

	if 'masked_pred' in keys:
		keys.remove('masked_pred')

	if plot_method=='boxplot':
		matplotlib.rc('font', **{'size': 10})

		fig, ax = plt.subplots(figsize=(6,4), dpi=DPI)

		new_df = pd.DataFrame({"perturbation":df["perturbation"], "blosum_distance":df['blosum_dist']}) 

		if distances_df_adversarial is not None:
			entry = pd.DataFrame({"blosum_distance":distances_df_adversarial['blosum_distance']}).assign(perturbation='other') 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		if distances_df_protherm is not None:
			entry = pd.DataFrame({"blosum_distance":distances_df_protherm['blosum_distance']}).assign(perturbation='protherm') 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		print(new_df)
		sns.boxplot(data=new_df, y='blosum_distance', x="perturbation", palette=palette)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 3.5), dpi=DPI)

		for idx, key in enumerate(keys):
			hist=False #True if key=='protherm' else False
			tmp_df = df[df['perturbation']==key]
			sns.distplot(x=tmp_df["blosum_dist"], label=key, kde=True, hist=hist,
				kde_kws={'linestyle':linestyles[idx]})

		if distances_df_adversarial is not None:
			sns.distplot(x=distances_df_adversarial['blosum_distance'], label='other', kde=True, hist=hist,
				kde_kws={'linestyle':linestyles[idx+1]})

		if distances_df_protherm is not None:
			sns.distplot(x=distances_df_protherm['blosum_distance'], label='protherm', kde=True, hist=True,
				kde_kws={'linestyle':linestyles[idx+2]})

		plt.xlabel(r'Blosum distance')
		plt.legend()

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_blosum_distances.png"))
		plt.close()

	return fig
