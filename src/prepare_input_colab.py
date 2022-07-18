import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
from collections import defaultdict
from Bio.PDB.PDBParser import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
import pdb


##############FUNCTIONS##############
def format_line(atm_no, atm_name, res_name, chain, res_no, x,y,z,occ,B,atm_id):
    '''Format the line into PDB
    '''

    #Get blanks
    atm_no = ' '*(5-len(atm_no))+atm_no
    atm_name = atm_name+' '*(4-len(atm_name))
    res_no = ' '*(4-len(res_no))+res_no
    x =' '*(8-len(x))+x
    y =' '*(8-len(y))+y
    z =' '*(8-len(z))+z
    occ = ' '*(6-len(occ))+occ
    B = ' '*(6-len(B))+B

    line = 'ATOM  '+atm_no+'  '+atm_name+res_name+' '+chain+res_no+' '*4+x+y+z+occ+B+' '*11+atm_id+'  '
    return line

def read_pdb(pdbname):
    '''Read PDB
    '''
    three_to_one = {'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU':'E', 'SER':'S', 'THR':'T', 'ASN':'N', 'GLN':'Q', 'CYS':'C', 'GLY':'G', 'PRO':'P', 'ALA':'A', 'ILE':'I', 'LEU':'L', 'MET':'M', 'PHE':'F', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
    'SEC':'U', 'PYL':'O', 'GLX':'X', 'UNK': 'X'}
    parser = PDBParser()
    struc = parser.get_structure('', pdbname)

    #Save
    cat_model = {}
    cat_model_resnos = {}
    cat_model_CA_coords = {}
    atm_no=0
    for model in struc:
        for chain in model:
            #Save
            cat_model[chain.id]=[]
            cat_model_resnos[chain.id]=[]
            cat_model_CA_coords[chain.id]=[]

            #Reset res no
            res_no=0
            for residue in chain:
                res_no +=1
                res_name = residue.get_resname()
                if res_name not in [*three_to_one.keys()]:
                    continue
                for atom in residue:
                    atm_no+=1
                    if atm_no>99999:
                        print('More than 99999 atoms',pdbname)
                        return {}
                    atom_id = atom.get_id()
                    atm_name = atom.get_name()
                    x,y,z = atom.get_coord()

                    if atm_name=='CA':
                        cat_model_CA_coords[chain.id].append(atom.get_coord())

                    x, y, z = format(x,'.3f'),format(y,'.3f'),format(z,'.3f')
                    occ = atom.get_occupancy()
                    B = min(100,atom.get_bfactor())
                    #Format line
                    line = format_line(str(atm_no), atm_name, res_name, chain.id, str(res_no),
                    x,y,z,str(occ),str(B),atom_id[0])
                    cat_model[chain.id].append(line+'\n')
                    cat_model_resnos[chain.id].append(resno)

    for key in cat_model:
        cat_model[key] = np.array(cat_model[key])
        cat_model_resnos[key] = np.array(cat_model_resnos[key])
        cat_model_CA_coords[key] = np.array(cat_model_CA_coords[key])

    return cat_model, cat_model_resnos, cat_model_CA_coords



def write_pdb(data, outname):
    '''Write PDB
    '''

    with open(outname, 'w') as file:
        for line in data:
            file.write(line)

def prepare_input(pdbname, receptor_chain, target_residues, COM, outdir):
    '''Prepare input
    '''

    #Read PDB
    cat_model, cat_model_resnos, cat_model_CA_coords = read_pdb(pdbname)
    receptor_pdb, receptor_resnos, receptor_CA_coords = cat_model[receptor_chain], cat_model_resnos[receptor_chain], cat_model_CA_coords[receptor_chain]
    #Write receptor for vis
    write_pdb(receptor_pdb, outdir+'receptor.pdb')
    #Write the target residues for vis
    target_residue_indices = []
    for i in range(len(receptor_resnos)):
        if receptor_resnos[i] in target_residues:
            target_residue_indices.append(i)

    target_residue_pdb = receptor_pdb[target_residue_indices]
    #Write receptor target residues for vis
    write_pdb(target_residue_pdb, outdir+'receptor_target_residues.pdb')

    return receptor_CAs
