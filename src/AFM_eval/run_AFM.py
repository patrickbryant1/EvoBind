# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import foldonly
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.model import modules_multimer
import jax
import jax.numpy as jnp
#from alphafold.relax import relax
import numpy as np
import glob
from collections import Counter
from scipy.special import softmax
import optax
import pandas as pd

import argparse
import sys
import os
import pdb


parser = argparse.ArgumentParser(description = '''"""Predicts structure using AlphaFold-multimer for the given receptor+peptide complex.
This script assumes features have already been generated.""".''')

parser.add_argument('--receptor_features', nargs=1, type= str, default=sys.stdin, help = 'Path to AFM features for the recptor.')
parser.add_argument('--data_dir', nargs=1, type= str, default=sys.stdin, help = 'Path to directory that contains the params.')
parser.add_argument('--model_preset', nargs=1, type= str, default=sys.stdin, help = 'multimer')
parser.add_argument('--peptide_sequences', nargs=1, type= str, default=sys.stdin, help = 'Peptide sequences (csv).')
parser.add_argument('--num_recycles', nargs=1, type= int, default=sys.stdin, help = 'Number of recycles.')
parser.add_argument('--cyclic_offset', nargs=1, type= int, default=sys.stdin, help = 'Use a cyclic offset for the peptide (1).')
parser.add_argument('--use_dropout', nargs=1, type= int, default=sys.stdin, help = 'Use dropout (1) or not (0).')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory.')


##########################FUNCTIONS##########################
def peptide_sequence_to_int(peptide_sequence):
    """Convert the peptide sequence to an int representation
    """

    restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])

    pep_int = []

    for c in peptide_sequence:
        pep_int.append(np.argwhere(restypes==c)[0][0])

    return np.array(pep_int,dtype='int32')

def peptide_sequence_to_all_atom_mask(peptide_sequence):
    """Convert the peptide sequence to an atomic representation
    where the mask (37) indicates what atoms are present

    GAVLITSMCPFYWHKRDENQ
    """

    mask_per_aa = {   'G': np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'A': np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'V': np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'L': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'I': np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'T': np.array([1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'S': np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'M': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'C': np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'P': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'F': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                      'Y': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
                      'W': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]),
                      'H': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'K': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
                      'R': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]),
                      'D': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'E': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'N': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      'Q': np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                  }
    pep_mask = []
    for c in peptide_sequence:
        pep_mask.append(mask_per_aa[c])

    return np.array(pep_mask)

def update_feats(feats, peptide_sequence):
    """Update the AFM feats to include the new peptide seq

    'aatype', #(receptor+peptide length) - update with peptide seq
    'residue_index', #(receptor+peptide length) - update with peptide (zero indexed)
    'seq_length', #(receptor+peptide length)
    'msa', #msa_length x (receptor+peptide length) - set to zero for peptide
    'num_alignments', #msa_length
    'template_aatype', #4x(receptor+peptide length)
    'template_all_atom_mask', #4x(receptor+peptide length)x37
    'template_all_atom_positions', #4x(receptor+peptide length)x37x3
    'asym_id', #(receptor+peptide length) - 1 for receptor 2 for peptide
    'sym_id', #(receptor+peptide length) - ones
    'entity_id', #(receptor+peptide length) - 1 for receptor 2 for peptide
    'deletion_matrix', #msa_length x (receptor+peptide length) - set to zero for peptide
    'deletion_mean', #(receptor+peptide length) - set to zero for peptide
    'all_atom_mask', #(receptor+peptide length)x37 - update for peptide according to sequence
    'all_atom_positions',  #(receptor+peptide length)x37x3 - all zeros
    'assembly_num_chains', #np.array(2)
    'entity_mask',  #(receptor+peptide length) - ones
    'num_templates', np.array(N) - leave as is
    'cluster_bias_mask', #MSA length - leave as is
    'bert_mask', #msa_length x (receptor+peptide length) - set to zero for peptide
    'seq_mask', #(receptor+peptide length) - ones
    'msa_mask' #msa_length x (receptor+peptide length) - set to zero for peptide
    """

    rl = len(feats['aatype']) #Length of receptor
    pl = len(peptide_sequence) #Length of peptide
    tot_len = rl+pl #Total length
    msa_len = len(feats['msa'])

    #Create arrays to be filled
    receptor_peptide_feats = {}
    #Peptide sequence to int
    int_pep_seq = peptide_sequence_to_int(peptide_sequence)
    receptor_peptide_feats['aatype'] = np.concatenate([feats['aatype'], int_pep_seq])
    receptor_peptide_feats['residue_index']=np.concatenate([feats['residue_index'], np.arange(pl)])
    receptor_peptide_feats['seq_length']=tot_len
    receptor_peptide_feats['msa']=np.concatenate([feats['msa'],np.zeros((msa_len,pl), dtype='int32')],axis=1)
    #Set first MSA row to seq!
    receptor_peptide_feats['msa'][0] = receptor_peptide_feats['aatype']
    #Set the rest to gaps
    receptor_peptide_feats['msa'][1:,rl:]=21
    receptor_peptide_feats['num_alignments']=msa_len
    receptor_peptide_feats['template_aatype']=np.concatenate([feats['template_aatype'],np.zeros((4,pl),dtype='int32')],axis=1)
    receptor_peptide_feats['template_all_atom_mask']=np.concatenate([feats['template_all_atom_mask'],np.zeros((4,pl,37),dtype='int32')],axis=1)
    receptor_peptide_feats['template_all_atom_positions']=np.concatenate([feats['template_all_atom_positions'],np.zeros((4,pl,37,3))],axis=1)
    receptor_peptide_feats['asym_id']=np.concatenate([feats['asym_id'], np.array([2]*pl)])
    receptor_peptide_feats['sym_id']=np.concatenate([feats['sym_id'], np.array([1]*pl)])
    receptor_peptide_feats['entity_id']=np.concatenate([feats['entity_id'], np.array([2]*pl)])
    receptor_peptide_feats['deletion_matrix']=np.concatenate([feats['deletion_matrix'],np.zeros((msa_len,pl))],axis=1)
    receptor_peptide_feats['deletion_mean']=np.concatenate([feats['deletion_mean'], np.zeros(pl)])
    pep_mask =  peptide_sequence_to_all_atom_mask(peptide_sequence) #Get atomic mask for peptide
    receptor_peptide_feats['all_atom_mask']=np.concatenate([feats['all_atom_mask'], pep_mask])
    receptor_peptide_feats['all_atom_positions']=np.zeros((tot_len,37,3))
    receptor_peptide_feats['assembly_num_chains']=np.array(2)
    receptor_peptide_feats['entity_mask']=np.ones(tot_len,dtype='int32')
    receptor_peptide_feats['num_templates']=feats['num_templates']
    receptor_peptide_feats['cluster_bias_mask']=feats['cluster_bias_mask']
    receptor_peptide_feats['bert_mask']=np.concatenate([feats['bert_mask'], np.ones((msa_len, pl))],axis=1)
    receptor_peptide_feats['seq_mask']=np.ones(tot_len)
    receptor_peptide_feats['msa_mask']=np.concatenate([feats['msa_mask'],np.ones((msa_len,pl))],axis=1)

    return receptor_peptide_feats

def predict_structure(
    features: str,
    output_dir: str,
    model_runners: Dict[str, model.RunModel],
    random_seed: int,
    peptide_sequences,
    cyclic_offset,
    config):
    """Predicts structure using AlphaFold for the given sequence."""


    #Load receptor features from AFM
    receptor_feats = np.load(features, allow_pickle=True)

    #Loop through all peptide seqs
    PAEs = []
    for ind, row in peptide_sequences.iterrows():
        #Path to output prediction
        unrelaxed_pdb_path = os.path.join(output_dir+'/', row.sequence+'.pdb')
        #If outfile exists - continue
        if os.path.exists(unrelaxed_pdb_path):
            continue
        #Update feats
        receptor_peptide_feats = update_feats(receptor_feats, row.sequence)

        #Loop through all model runners
        for model_index, (model_name, model_runner) in enumerate(model_runners.items()):

            #Process feats
            model_random_seed = model_index + random_seed * ind
            processed_feature_dict = model_runner.process_features(
                receptor_peptide_feats, random_seed=model_random_seed)

            #Add cyclic
            if cyclic_offset==True:
                peptide_length = len(row.sequence) #The length may vary
                peptide_cyclic_offset_array = np.zeros((peptide_length, peptide_length))
                cyc_row = np.arange(0,-peptide_length,-1)
                pc = int(np.round(peptide_length/2)) #Get centre
                cyc_row[pc+1:]=np.arange(len(cyc_row[pc+1:]),0,-1)
                for i in range(len(peptide_cyclic_offset_array)):
                    peptide_cyclic_offset_array[i]=np.roll(cyc_row,i)
                #Update the entire positional array
                pos = processed_feature_dict['residue_index']
                cyclic_offset_array = pos[:, None] - pos[None, :]
                cyclic_offset_array[-len(peptide_cyclic_offset_array):,-len(peptide_cyclic_offset_array):]=peptide_cyclic_offset_array
                processed_feature_dict['cyclic_offset']=cyclic_offset_array

            #Predict
            print('Predicting', row.sequence)
            prediction_result = model_runner.predict(processed_feature_dict) #The params for model runner are contained within self
            # Add the predicted LDDT in the b-factor column.
            plddt_per_pos = prediction_result['plddt']
            plddt_b_factors = np.repeat(plddt_per_pos[:, None], residue_constants.atom_type_num, axis=-1)
            #Get the unrelaxed protein
            unrelaxed_protein = protein.from_prediction(features=processed_feature_dict, result=prediction_result,  b_factors=plddt_b_factors,
                                                        remove_leading_feature_dimension=not model_runner.multimer_mode)
            unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
            #Save
            with open(unrelaxed_pdb_path, 'w') as f:
                f.write(unrelaxed_pdb)

            print('Saved',unrelaxed_pdb_path)

            #Get PAE
            pl = len(prediction_result['predicted_aligned_error'])-len(row.sequence)
            pae = float(np.mean(prediction_result['predicted_aligned_error'][:pl,pl:]))
            PAEs.append(pae)

    #Save PAEs
    pae_df = peptide_sequences
    pae_df['PAE'] = PAEs
    pae_df.to_csv(output_dir+'interface_pae.csv', index=None)


##################MAIN#######################
########Here the model runners are setup and the predict function is called##########

#Parse args
args = parser.parse_args()
peptide_sequences = pd.read_csv(args.peptide_sequences[0])
#Define model
model_runners = {}
model_names = config.MODEL_PRESETS[args.model_preset[0]]
model_name = model_names[0] #Only use the first model here
model_config = config.model_config(model_name)
#Set the dropout in the str module to 0 -  from AFsample: https://github.com/bjornwallner/alphafoldv2.2.0/blob/main/run_alphafold.py
model_config.model.heads.structure_module.dropout=0.0
model_config.model.num_ensemble_eval = 1 #Use 1 ensemble
model_config.model.num_ensemble_train = 1 #Use 1 ensemble
model_config.model.num_recycle = args.num_recycles[0]
model_params = data.get_model_haiku_params(
model_name=model_name, data_dir=args.data_dir[0])

#Use dropout by default
try:
    use_dropout = bool(args.use_dropout[0])
except:
    use_dropout = True

model_runner = model.RunModel(model_config, model_params, is_training=use_dropout) #Set training to true to have dropout in the Evoformer
model_runners[f'{model_name}_pred_{0}'] = model_runner

#This is used in the feature processing
#For multimers, this is simply returned though: if self.multimer_mode:
                                            #    return raw_features
#This is because the sampling happens within the multimer runscript
#A new random key is fetched each iter
random_seed = random.randrange(sys.maxsize // len(model_runners))

#Check if cyclic design
try:
    cyclic_offset = bool(args.cyclic_offset[0])
except:
    cyclic_offset = False

#Update config
if cyclic_offset==True:
    model_config.model.embeddings_and_evoformer.cyclic_offset=True
else:
    model_config.model.embeddings_and_evoformer.cyclic_offset=None



# Predict structure
predict_structure(
    features=args.receptor_features[0],
    output_dir=args.outdir[0],
    model_runners=model_runners,
    random_seed=random_seed,
    peptide_sequences=peptide_sequences,
    cyclic_offset=cyclic_offset,
    config=model_config)
