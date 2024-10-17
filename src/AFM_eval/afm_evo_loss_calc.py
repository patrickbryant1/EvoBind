import os
import pickle
import sys
import time
import argparse
import pandas as pd
import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SVDSuperimposer import SVDSuperimposer

import pdb




parser = argparse.ArgumentParser(description = """Score AFM vs EvoBind.""")

parser.add_argument('--afm_dir', nargs=1, type= str, default=sys.stdin, help = 'Path to location of AFM preds.')
parser.add_argument('--evo_dir', nargs=1, type= str, default=sys.stdin, help = 'Path to location of EvoBind preds.')
parser.add_argument('--score_csv', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with sequences and ids to be scores.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

##############FUNCTIONS##############

def read_pdb(pdbname):
    '''Read PDB
    '''

    f=open(pdbname,'rt')

    if '.pdb' in pdbname:
        parser = PDBParser()
        struc = parser.get_structure('', f)
    else:
        parser = MMCIFParser()
        struc = parser.get_structure('',f)

    #Save
    model_coords = {}
    model_3seq = {}
    model_resnos = {}
    model_atoms = {}
    model_bfactors = {}


    for model in struc:
        for chain in model:
            #Save
            model_coords[chain.id]=[]
            model_3seq[chain.id]=[]
            model_resnos[chain.id]=[]
            model_atoms[chain.id]=[]
            model_bfactors[chain.id]=[]

            #Go through al residues
            for residue in chain:
                res_name = residue.get_resname()
                if is_aa(residue)!=True:
                    continue
                for atom in residue:
                    atom_id = atom.get_id()
                    atm_name = atom.get_name()
                    #Save
                    model_coords[chain.id].append(atom.get_coord())
                    model_3seq[chain.id].append(res_name)
                    model_resnos[chain.id].append(residue.get_id()[1])
                    model_atoms[chain.id].append(atom_id)
                    model_bfactors[chain.id].append(atom.bfactor)



    return model_coords, model_3seq, model_resnos, model_atoms, model_bfactors

def score_structures(afm_dir, evo_dir, score_df, outdir):
    """Calcualte losses
    """
    
    
    loss_df = {'iter':[], 'sequence':[], 'loss':[], 'plddt':[], 'COM':[]}

    #Superimposer
    sup = SVDSuperimposer()

    for ind, row in score_df.iterrows():
        if row.iteration=='init':
            continue

        try:
            #Read EvoBind prediction
            evo_coords, evo_3seq, evo_resnos, evo_atoms, evo_bfactors = read_pdb(evo_dir+'unrelaxed_'+str(row.iteration)+'.pdb')
        except:
            print('Could not read file:',evo_dir+'unrelaxed_'+str(row.iteration)+'.pdb')
            #sys.exit()
            continue
        #Read AFM prediction
        afm_coords, afm_3seq, afm_resnos, afm_atoms, afm_bfactors = read_pdb(afm_dir+row.sequence+'.pdb')
        #Get chains - first receptor, second peptide
        evo_chains = [*evo_coords.keys()]
        afm_chains = [*afm_coords.keys()]

        #Define coords
        evo_rec_coords, evo_pep_coords = np.array(evo_coords[evo_chains[0]]), np.array(evo_coords[evo_chains[1]])
        afm_rec_coords, afm_pep_coords = np.array(afm_coords[afm_chains[0]]), np.array(afm_coords[afm_chains[1]])

        #Align CAs
        evo_rec_CAs = evo_rec_coords[np.argwhere(np.array(evo_atoms[evo_chains[0]])=='CA')[:,0]]
        afm_rec_CAs = afm_rec_coords[np.argwhere(np.array(afm_atoms[afm_chains[0]])=='CA')[:,0]]
        sup.set(evo_rec_CAs, afm_rec_CAs) #(reference_coords, coords)
        sup.run()
        rot, tran = sup.get_rotran()
        #Rotate the peptide coords to match the centre of mass for the native comparison
        evo_pep_CAs = evo_pep_coords[np.argwhere(np.array(evo_atoms[evo_chains[1]])=='CA')[:,0]] 
        afm_pep_CAs = afm_pep_coords[np.argwhere(np.array(afm_atoms[afm_chains[1]])=='CA')[:,0]]
        rotated_afm_pep_CAs = np.dot(afm_pep_CAs, rot) + tran
        #Peptide COM
        evo_pep_CM = np.sum(evo_pep_CAs,axis=0)/evo_pep_CAs.shape[0]
        afm_pep_CM = np.sum(rotated_afm_pep_CAs,axis=0)/rotated_afm_pep_CAs.shape[0]
        delta_CM = np.sqrt(np.sum(np.square(evo_pep_CM-afm_pep_CM)))

        #Get EvoBind if pos - CBs<8Å
        mat = np.append(evo_rec_coords[np.argwhere(np.array(evo_atoms[evo_chains[0]])=='CB')[:,0]], evo_pep_coords[np.argwhere(np.array(evo_atoms[evo_chains[1]])=='CB')[:,0]],axis=0)
        a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
        evo_dmat = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
        l1 = len(np.argwhere(np.array(evo_atoms[evo_chains[0]])=='CB')[:,0])
        #Get interface
        contact_dists = evo_dmat[:l1,l1:] #first dimension=receptor, second=peptide
        #Get receptor contact positions - the indices will match between the predictions (no gaps in pred receptors)
        rec_if_pos = np.unique(np.argwhere(contact_dists<8)[:,0])
        if_resno_pos = []
        for ri in rec_if_pos:
            if_resno_pos.extend([*np.argwhere(evo_resnos[evo_chains[0]]==ri+1)[:,0]])

        #Get AFM interface dists
        if len(if_resno_pos)>0:
            mat = np.append(afm_rec_coords[np.array(if_resno_pos)], afm_pep_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            afm_dmat = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            l1 = len(if_resno_pos)
            afm_if = afm_dmat[:l1,l1:] #first dimension=receptor, second=peptide

            #Get the closest atom-atom distances across the receptor interface residues.
            afm_rec_avg_if_dist = afm_if[np.arange(afm_if.shape[0]), np.argmin(afm_if,axis=1)].mean()
            afm_pep_avg_if_dist = afm_if[np.argmin(afm_if,axis=0),np.arange(afm_if.shape[1])].mean()
        else:
            afm_rec_avg_if_dist = 20
            afm_pep_avg_if_dist = 20
        #Get the peptide plDDT
        afm_pep_plddt = np.mean(afm_bfactors[afm_chains[1]])
        #Calc AFM-EvoBind loss
        loss = (afm_rec_avg_if_dist+afm_pep_avg_if_dist)/2*1/afm_pep_plddt*delta_CM
        
        
        #Save 
        loss_df['iter'].append(str(row.iteration))
        loss_df['sequence'].append(row.sequence)
        loss_df['loss'].append(loss)
        loss_df['plddt'].append(afm_pep_plddt)
        loss_df['COM'].append(delta_CM)
    
    #Save
    loss_df = pd.DataFrame(loss_df)
    loss_df.to_csv(outdir+'afm_evo_metrics.csv',index=None)
    print('Saved losses to',outdir+'afm_evo_metrics.csv')

##################MAIN#####################

#Parse args
args = parser.parse_args()
afm_dir = args.afm_dir[0]
evo_dir = args.evo_dir[0]
score_df = pd.read_csv(args.score_csv[0])
outdir = args.outdir[0]

score_structures(afm_dir, evo_dir, score_df, outdir)
