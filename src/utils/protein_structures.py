import Bio
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
import numpy as np
import pandas as pd

print("Biopython v" + Bio.__version__)


def get_coordinates(protein_name, pdb_filename):

    p = PDBParser()
    s = p.get_structure(protein_name, pdb_filename) 

    coordinates = []

    for chains in s:
        for chain in chains:
            for residue in chain:    
                for atom in residue:
                    if atom.name=="CA":
                        coordinates.append(atom.get_coord())

    coordinates = np.array(coordinates)
    return coordinates

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

def get_RMSD(coordinates_1, coordinates_2):
    d_i = np.linalg.norm(coordinates_1 - coordinates_2, axis=1)
    rmsd = np.sqrt((d_i**2).mean())
    assert rmsd > 0
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
    lddt = np.mean(preserved_fractions)
    if precision > 0:
        lddt = round(lddt, precision)

    assert lddt > 0 and lddt < 1
    return lddt

def get_TM_score(original_length, coordinates_1, coordinates_2):

    L_N = original_length
    d_i = np.linalg.norm(coordinates_1 - coordinates_2, axis=1)
    d_0 = 1.24*np.cbrt(L_N-15)-1.8

    tm_score = (1 / (1 + (d_i**2 / d_0**2))).sum()/L_N

    assert tm_score > 0 and tm_score < 1
    return tm_score
