import re
import sys
import random
import os.path
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from google.colab import files

import Bio
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser

from colabfold.utils import setup_logging
from colabfold.colabfold import plot_protein
from colabfold.batch import get_queries, run, set_model_type
from colabfold.download import download_alphafold_params, default_data_dir


def predict_structure(name, query_sequence, savedir, filename, alphafold_dir="alphafold/"):

    out_dir = os.path.join(savedir, "structures/", filename+"/")
    alphafold_dir = os.path.join(savedir, "alphafold/")

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    os.makedirs(os.path.dirname(alphafold_dir), exist_ok=True)

    def add_hash(x,y):
        return x+"_"+hashlib.sha1(y.encode()).hexdigest()[:5]

    jobname = name
    query_sequence = "".join(query_sequence.split())
    basejobname = "".join(jobname.split())
    basejobname = re.sub(r'\W+', '', basejobname)
    queries_path=f"{out_dir}{jobname}.csv"

    with open(queries_path, "w") as text_file:
        text_file.write(f"id,sequence\n{jobname},{query_sequence}")

    ### msa

    msa_mode = "MMseqs2 (UniRef+Environmental)" #@param ["MMseqs2 (UniRef+Environmental)", "MMseqs2 (UniRef only)","single_sequence","custom"]
    pair_mode = "unpaired+paired" #@param ["unpaired+paired","paired","unpaired"] {type:"string"}

    # decide which a3m to use
    if msa_mode.startswith("MMseqs2"):
        a3m_file = f"{out_dir}{jobname}.a3m"
    elif msa_mode == "custom":
        a3m_file = f"{out_dir}{jobname}.custom.a3m"
        if not os.path.isfile(a3m_file):
            custom_msa_dict = files.upload()
            custom_msa = list(custom_msa_dict.keys())[0]
            header = 0
            import fileinput
            for line in fileinput.FileInput(custom_msa,inplace=1):
                if line.startswith(">"):
                    header = header + 1
                if not line.rstrip():
                    continue
                if line.startswith(">") == False and header == 1:
                    query_sequence = line.rstrip()
                print(line, end='')

        os.rename(custom_msa, a3m_file)
        queries_path=a3m_file
        print(f"moving {custom_msa} to {a3m_file}")
    else:
        a3m_file = f"{out_dir}{jobname}.single_sequence.a3m"
        with open(a3m_file, "w") as text_file:
            text_file.write(">1\n%s" % query_sequence)

    ### settings

    model_type = "AlphaFold2-ptm" #@param ["auto", "AlphaFold2-ptm", "AlphaFold2-multimer-v1", "AlphaFold2-multimer-v2"]
    num_recycles = 3 #@param [1,3,6,12,24,48] {type:"raw"}
    dpi = 200 #@param {type:"integer"}

    ### run prediction

    if "TF_FORCE_UNIFIED_MEMORY" in os.environ:
        del os.environ["TF_FORCE_UNIFIED_MEMORY"]
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
        del os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]

    def prediction_callback(unrelaxed_protein, length, prediction_result, input_features, type):
        fig = plot_protein(unrelaxed_protein, Ls=length, dpi=150)
        fig.savefig(os.path.join(out_dir, jobname+"_structure.png"))
        plt.close()

    result_dir=out_dir
    setup_logging(Path(alphafold_dir).joinpath("log.txt"))
    queries, is_complex = get_queries(queries_path)
    model_type = set_model_type(is_complex, model_type)
    download_alphafold_params(model_type, Path(alphafold_dir))
    run(
        queries=queries,
        result_dir=result_dir,
        use_templates=False,
        custom_template_path=None,
        use_amber=True, 
        msa_mode=msa_mode,    
        model_type=model_type,
        num_models=1,
        num_recycles=num_recycles,
        model_order=[1],#, 2, 3, 4, 5],
        is_complex=is_complex,
        data_dir=Path(alphafold_dir),
        keep_existing_results=False,
        recompile_padding=1.0,
        rank_by="auto",
        pair_mode=pair_mode,
        stop_at_score=float(100),
        prediction_callback=prediction_callback,
        dpi=dpi
    )


def get_coordinates(protein_name, pdb_filename):

    p = PDBParser()
    s = p.get_structure(protein_name, pdb_filename) 

    coordinates = []

    count = 1
    residue_offset = 0
    missing_residues = []

    for chains in s:
        for chain in chains:
            for residue in chain:    
                for atom in residue:

                    if atom.name=="CA":
                        residue_idx = residue._id[1]-residue_offset

                        if residue_idx!=count:
                            print(f"\n\t{protein_name} missing {count}-th Ca atom")
                            residue_offset += 1
                            missing_residues.append(count)

                        coordinates.append(atom.get_coord())
                        count += 1

    coordinates = np.array(coordinates)
    return coordinates, missing_residues

def get_corresponding_residues_coordinates(protein_name_1, pdb_filename_1, protein_name_2, pdb_filename_2):

    p = PDBParser()
    s1 = p.get_structure(protein_name_1, pdb_filename_1) 
    s2 = p.get_structure(protein_name_2, pdb_filename_2) 

    coordinates1 = []
    coordinates2 = []

    for chains1, chains2 in zip(s1,s2):
        for chain1, chain2 in zip(chains1,chains2):
            for residue1, residue2 in zip(chain1, chain2):
                if residue1.resname==residue2.resname:
                    for atom1,atom2 in zip(residue1,residue2):
                        if atom1.name=="CA" and atom2.name=="CA":
                            coordinates1.append(atom1.get_coord())
                            coordinates2.append(atom2.get_coord())

    coordinates1 = np.array(coordinates1)
    coordinates2 = np.array(coordinates2)
    return coordinates1, coordinates2

def get_available_residues_coordinates(protein_name_1, pdb_filename_1, protein_name_2, pdb_filename_2):

    p = PDBParser()
    s1 = p.get_structure(protein_name_1, pdb_filename_1) 
    s2 = p.get_structure(protein_name_2, pdb_filename_2) 

    coordinates1 = []
    coordinates2 = []

    for chains1, chains2 in zip(s1,s2):
        for chain1, chain2 in zip(chains1,chains2):
            for residue1, residue2 in zip(chain1, chain2):
                if residue1._id[1]==residue2._id[1]:
                    for atom1,atom2 in zip(residue1,residue2):
                        if atom1.name=="CA" and atom2.name=="CA":
                            coordinates1.append(atom1.get_coord())
                            coordinates2.append(atom2.get_coord())

    coordinates1 = np.array(coordinates1)
    coordinates2 = np.array(coordinates2)
    assert coordinates1.shape == coordinates2.shape
    return coordinates1, coordinates2

def get_RMSD(coordinates_1, coordinates_2):
    d_i = np.linalg.norm(coordinates_1 - coordinates_2, axis=1)
    rmsd = np.sqrt((d_i**2).mean())
    assert rmsd >= 0
    return rmsd

def get_dmap(cb_coordinates):
    L = len(cb_coordinates)
    cb_map = np.full((L, L), np.nan)
    for col1_idx, r1 in enumerate(cb_coordinates):
        (a, b, c) = r1
        for col2_idx, r2 in enumerate(cb_coordinates):
            (p, q, r) = r2
            cb_map[col1_idx, col2_idx] = np.sqrt((a-p)**2+(b-q)**2+(c-r)**2)
    return cb_map

def get_LDDT(true_dmap, pred_dmap, cutoff=15, sep_thresh=-1, tolerances=[0.5, 1, 2, 4], precision=4):

    def get_flattened(dmap):
      if dmap.ndim == 1:
        return dmap
      elif dmap.ndim == 2:
        return dmap[np.triu_indices_from(dmap, k=1)]
      else:
        assert False, "ERROR: the passes array has dimension not equal to 2 or 1!"

    def get_separations(dmap):
        t_indices = np.triu_indices_from(dmap, k=1)
        separations = np.abs(t_indices[0] - t_indices[1])
        return separations

    # return a 1D boolean array indicating where the sequence separation in the
    # upper triangle meets the threshold comparison
    def get_sep_thresh_b_indices(dmap, thresh, comparator):
        assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
        dmap_flat = get_flattened(dmap)
        separations = get_separations(dmap)
        if comparator == 'gt':
            threshed = separations > thresh
        elif comparator == 'lt':
            threshed = separations < thresh
        elif comparator == 'ge':
            threshed = separations >= thresh
        elif comparator == 'le':
            threshed = separations <= thresh
        return threshed

    # return a 1D boolean array indicating where the distance in the
    # upper triangle meets the threshold comparison
    def get_dist_thresh_b_indices(dmap, thresh, comparator):
        assert comparator in {'gt', 'lt', 'ge', 'le'}, "ERROR: Unknown comparator for thresholding!"
        dmap_flat = get_flattened(dmap)
        if comparator == 'gt':
            threshed = dmap_flat > thresh
        elif comparator == 'lt':
            threshed = dmap_flat < thresh
        elif comparator == 'ge':
            threshed = dmap_flat >= thresh
        elif comparator == 'le':
            threshed = dmap_flat <= thresh
        return threshed

    def get_n_preserved(new_flat_dmap, original_flat_dmap, thresh):
        err = np.abs(new_flat_dmap - original_flat_dmap)
        n_preserved = (err < thresh).sum()
        return n_preserved

    # flatten upper triangles
    true_flat_map = get_flattened(true_dmap)
    pred_flat_map = get_flattened(pred_dmap)

    # find set L
    S_thresh_indices = get_sep_thresh_b_indices(true_dmap, sep_thresh, 'gt')
    R_thresh_indices = get_dist_thresh_b_indices(true_flat_map, cutoff, 'lt')
    L_indices = S_thresh_indices & R_thresh_indices
    true_flat_in_L = true_flat_map[L_indices]
    pred_flat_in_L = pred_flat_map[L_indices]

    # number of pairs in L
    L_n = L_indices.sum()

    preserved_fractions = []
    for tol in tolerances:
        _n_preserved = get_n_preserved(true_flat_in_L, pred_flat_in_L, tol)
        _f_preserved = _n_preserved / L_n
        preserved_fractions.append(_f_preserved)
    lddt = np.mean(preserved_fractions)*100
    if precision > 0:
        lddt = round(lddt, precision)

    assert lddt >= 0 and lddt <= 100
    return lddt

def get_TM_score(original_length, coordinates_1, coordinates_2):

    L_N = original_length
    d_i = np.linalg.norm(coordinates_1 - coordinates_2, ord=2, axis=1)
    d_0 = 1.24*np.cbrt(L_N-15)-1.8

    tm_score = (1 / (1 + (d_i**2 / d_0**2))).sum()/L_N

    assert tm_score > 0 and tm_score <= 1
    return tm_score
