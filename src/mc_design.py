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

"""Full AlphaFold protein structure prediction script.
Modified by Patrick Bryant to include MC search.
"""
import pdb
import json
import os
import warnings
import pathlib
import pickle
import random
import sys

sys.path.append(os.getcwd()+"/src/AF2/")

import time
from typing import Dict, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import msaonly
from alphafold.data import foldonly
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
from jax import grad, value_and_grad
import copy
from collections import defaultdict
from Bio.SVDSuperimposer import SVDSuperimposer
from mutate_sequence import mutate_sequence


# Internal import (7716).

##### flags #####
flags.DEFINE_string('receptor_fasta_path', None,
    'Paths to FASTA file for receptor'
    'basename is used to name the output directories for '
    'each prediction.')

flags.DEFINE_string('receptor_if_residues', None,
    'Path to numpy array with receptor interface residues.')

flags.DEFINE_string('receptor_CAs', None,
    'Path to numpy array with receptor interface residues.')

flags.DEFINE_integer('peptide_length', None,
    'Length of peptide binder.')

flags.DEFINE_string('peptide_CM', None,
    'Required centre of mass for peptide to be designed.')

flags.DEFINE_list('model_names', None,
    'Comma separated list of different MSA sampling schemes to use; '
    'look at config for available options; will generate one output '
    'for each specified scheme.')

flags.DEFINE_string('output_dir', None,
    'Path to a directory that will store the results.')

flags.DEFINE_integer('max_recycles', 1,
    'Number of recyles through the model')

flags.DEFINE_list('msas', None,
    'Comma separated list of msa paths')

flags.DEFINE_integer('random_seed', None,
    'The random seed for the data pipeline. By default, this is randomly generated. '
    'Note that even if this is set, Alphafold may still not be deterministic, '
    'because processes like GPU inference are nondeterministic.')

flags.DEFINE_integer('num_iterations', None,
    'Number of iterations to run.')

flags.DEFINE_bool('predict_only', None,
    'Only predict, do not optimise.')

flags.DEFINE_bool('plDDT_only', None,
    'Use plDDT as the only loss.')

flags.DEFINE_string('peptide_sequence', None,
    'Only predict using this peptide sequence, do not optimise.')

flags.DEFINE_integer('cyclic_offset', None,
    'Use a cyclic offset for the peptide (1).')

##### databases flags #####
flags.DEFINE_string('data_dir', None,
    'Path to directory of supporting data.')

FLAGS = flags.FLAGS

############FUNCTIONS###########
def initialize_weights(peptide_length):
    '''Initialize sequence probabilities
    '''

    weights = np.random.gumbel(0,1,(peptide_length,20))
    weights = np.array([np.exp(weights[i])/np.sum(np.exp(weights[i])) for i in range(len(weights))])

    #Get the peptide sequence
    #Residue types
    restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])

    peptide_sequence = ''.join(restypes[[x for x in np.argmax(weights,axis=1)]])

    return weights, peptide_sequence

def create_weights_for_seq(peptide_sequence):
    '''Create wights for a certain peptide sequence
    '''

    restypes = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V' ])

    weights = np.zeros((len(peptide_sequence),20))

    for i in range(len(peptide_sequence)):
        weights[i,np.argwhere(restypes==peptide_sequence[i])[0][0]]=1

    return weights

def update_features(feature_dict, peptide_sequence):
    '''Update the features used for the pred:
    #Sequence features
    'aatype', 'between_segment_residues', 'domain_name',
    'residue_index', 'seq_length', 'sequence',
    #MSA features
    'deletion_matrix_int', 'msa', 'num_alignments',
    #Template features
    'template_aatype', 'template_all_atom_masks',
    'template_all_atom_positions', 'template_domain_names',
    'template_sequence', 'template_sum_probs'
    '''
    #Save
    new_feature_dict = {}

    peptide_features = pipeline.make_sequence_features(sequence=peptide_sequence,
                                                    description='peptide', num_res=len(peptide_sequence))
    #Merge sequence features
    #aatype
    new_feature_dict['aatype']=np.concatenate((feature_dict['aatype'],peptide_features['aatype']))
    #between_segment_residues
    new_feature_dict['between_segment_residues']=np.concatenate((feature_dict['between_segment_residues'],peptide_features['between_segment_residues']))
    #domain_name
    new_feature_dict['domain_name'] = feature_dict['domain_name']
    #residue_index
    new_feature_dict['residue_index']=np.concatenate((feature_dict['residue_index'],peptide_features['residue_index']+feature_dict['residue_index'][-1]+201))
    #seq_length
    new_feature_dict['seq_length']=np.concatenate((feature_dict['seq_length']+peptide_features['seq_length'][0],
                                            peptide_features['seq_length']+feature_dict['seq_length'][0]))
    #sequence
    new_feature_dict['sequence']=np.array(feature_dict['sequence'][0]+peptide_features['sequence'][0], dtype='object')

    #Merge MSA features
    #deletion_matrix_int
    new_feature_dict['deletion_matrix_int']=np.concatenate((feature_dict['deletion_matrix_int'],
                                            np.zeros((feature_dict['deletion_matrix_int'].shape[0],len(peptide_sequence)))), axis=1)
    #msa
    peptide_msa = np.zeros((feature_dict['deletion_matrix_int'].shape[0],len(peptide_sequence)),dtype='int32')
    peptide_msa[:,:] = 21
    HHBLITS_AA_TO_ID = {'A': 0,'B': 2,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'J': 20,'K': 8,'L': 9,'M': 10,'N': 11,
                        'O': 20,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'U': 1,'V': 17,'W': 18,'X': 20,'Y': 19,'Z': 3,'-': 21,}
    for i in range(len(peptide_sequence)):
        peptide_msa[0,i]=HHBLITS_AA_TO_ID[peptide_sequence[i]]
    new_feature_dict['msa']=np.concatenate((feature_dict['msa'], peptide_msa), axis=1)
    #num_alignments
    new_feature_dict['num_alignments']=np.concatenate((feature_dict['num_alignments'], feature_dict['num_alignments'][:len(peptide_sequence)]))

    #Merge template features
    for key in ['template_aatype', 'template_all_atom_masks', 'template_all_atom_positions',
                'template_domain_names', 'template_sequence', 'template_sum_probs']:
        new_feature_dict[key]=feature_dict[key]

    return new_feature_dict


def predict_function(peptide_sequence, feature_dict, output_dir, model_runners,
                     random_seed, receptor_if_residues, receptor_CAs, peptide_CM,  predict_only):
    '''Predict and calculate loss
    '''


    #Add features for the binder
    #Update features
    new_feature_dict = update_features(feature_dict, peptide_sequence)

    # Run the model.
    for model_name, model_runner in model_runners.items():
      #logging.info('Running model %s', model_name)
      processed_feature_dict = model_runner.process_features(
          new_feature_dict, random_seed=random_seed)

      if FLAGS.cyclic_offset:
          pos = new_feature_dict['residue_index']
          cyclic_offset_array = pos[:, None] - pos[None, :]
          peptide_cyclic_offset_array = feature_dict['peptide_cyclic_offset_array']
          cyclic_offset_array[-len(peptide_cyclic_offset_array):,-len(peptide_cyclic_offset_array):]=peptide_cyclic_offset_array
          processed_feature_dict['cyclic_offset']=np.expand_dims(cyclic_offset_array,axis=0)

      t_0 = time.time()
      prediction_result = model_runner.predict(processed_feature_dict)
      print('Prediction took', time.time() - t_0,'s')

    #Calculate loss
    #Loss features
    # Get the pLDDT confidence metric.
    plddt = prediction_result['plddt']
    #Get the protein
    plddt_b_factors = np.repeat(plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(features=processed_feature_dict,result=prediction_result,b_factors=plddt_b_factors)

    #If  predict_only - return unrelaxed_protein
    if predict_only==True:
        return 0, 0 ,0 , 0, unrelaxed_protein
    else:
        protein_resno, protein_atoms, protein_atom_coords = protein.get_coords(unrelaxed_protein)
        peptide_length = len(peptide_sequence)
        #Get residue index
        residue_index = new_feature_dict['residue_index']
        receptor_res_index = residue_index[:-peptide_length]
        peptide_res_index = residue_index[-peptide_length:]
        #Get coords
        receptor_coords = protein_atom_coords[np.argwhere(protein_resno<=receptor_res_index[-1]+1)[:,0]]
        peptide_coords = protein_atom_coords[np.argwhere(protein_resno>receptor_res_index[-1]+1)[:,0]]
        #Get atom types
        receptor_atoms = protein_atoms[np.argwhere(protein_resno<=receptor_res_index[-1]+1)[:,0]]
        peptide_atoms = protein_atoms[np.argwhere(protein_resno>receptor_res_index[-1]+1)[:,0]]
        #Get resno for each atom
        #Start at 1 - same for receptor_if_residues
        receptor_resno = protein_resno[np.argwhere(protein_resno<=receptor_res_index[-1]+1)[:,0]]
        peptide_resno = protein_resno[np.argwhere(protein_resno>peptide_res_index[0])[:,0]]
        #Get atoms belonging to if res for the receptor
        receptor_if_pos = []
        for ifr in receptor_if_residues:
            receptor_if_pos.extend([*np.argwhere(receptor_resno==ifr)])
        receptor_if_pos = np.array(receptor_if_pos)[:,0]

        #Calc 2-norm - distance between peptide and interface
        mat = np.append(peptide_coords,receptor_coords[receptor_if_pos],axis=0)
        a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
        dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
        l1 = len(peptide_coords)
        #Get interface
        contact_dists = dists[:l1,l1:] #first dimension = peptide, second = receptor

        #Get the closest atom-atom distances across the receptor interface residues.
        closest_dists_peptide = contact_dists[np.arange(contact_dists.shape[0]),np.argmin(contact_dists,axis=1)]
        closest_dists_receptor = contact_dists[np.argmin(contact_dists,axis=0),np.arange(contact_dists.shape[1])]

        #Get the peptide plDDT
        peptide_plDDT = plddt[-peptide_length:]

        #Superpose the receptor CAs and compare the centre of mass
        sup = SVDSuperimposer()

        #Get the CAs for the receptor and peptide: order N, CA
        pred_receptor_CAs, pred_peptide_CAs = [], []
        for resno in  np.unique(receptor_resno):
            pred_receptor_CAs.append(np.argwhere(receptor_resno==resno)[1][0])
        for resno in  np.unique(peptide_resno):
            pred_peptide_CAs.append(np.argwhere(peptide_resno==resno)[1][0])
        sup.set(receptor_CAs, receptor_coords[pred_receptor_CAs]) #(reference_coords, coords)
        sup.run()
        rot, tran = sup.get_rotran()
        #Rotate the peptide coords to match the centre of mass for the native comparison
        rotated_coords = np.dot(peptide_coords[pred_peptide_CAs], rot) + tran
        rotated_CM =  np.sum(rotated_coords,axis=0)/(rotated_coords.shape[0])
        delta_CM = np.sqrt(np.sum(np.square(peptide_CM-rotated_CM)))

        return closest_dists_peptide.mean(), closest_dists_receptor.mean(), peptide_plDDT.mean(), delta_CM, unrelaxed_protein

def parse_atm_record(line):
    '''Get the atm record
    '''
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11])
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26])
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record

def save_design(unrelaxed_protein, output_dir, model_name, l1):
    '''Save the resulting protein-peptide design to a pdb file
    '''

    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    chain_name = 'A'
    with open(unrelaxed_pdb_path, 'w') as f:
        pdb_contents = protein.to_pdb(unrelaxed_protein).split('\n')
        for line in pdb_contents:
            try:
                record = parse_atm_record(line)
                if record['res_no']>l1:
                    chain_name='B'
                outline = line[:21]+chain_name+line[22:]
                f.write(outline+'\n')
            except:
                f.write(line+'\n')


def optimise_binder(
    fasta_path: str,
    fasta_name: str,
    receptor_if_residues: str,
    receptor_CAs: str,
    peptide_length: int,
    peptide_CM: str,
    output_dir: str,
    data_pipeline: pipeline.DataPipeline,
    random_seed: int,
    model_runners: Optional[Dict[str, model.RunModel]],
    num_iterations: int,
    predict_only: bool,
    predict_only_sequence: str,
    plDDT_only: bool):

  """
  1. Initialize an array with rabdomly distributed sequence probabilities: initialize_sequence
  - This is the peptide binder sequence
  2. Make all input features to AlphaFold: update_features
  3. Predict the structure: model_runner.predict(processed_feature_dict)
  4. Mutate the sequence.
  5. Score the current peptide based on the distance from the peptide atoms to the interface, the
  inteface to the peptide atoms, the centre of mass of the peptide and the plDDT.
  6. Except the new sequence as a starting point if the if_dist is lower.
  7. Return to step 4.
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Get features.
  feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        input_msas=FLAGS.msas,
        template_search=None,
        msa_output_dir=None)




  #Initialize weights - these are the amino acid probabilities
  #Also returns the peptide_sequence corresponding to the weights
  seq_weights, peptide_sequence = initialize_weights(peptide_length)

  #Add cyclic
  if FLAGS.cyclic_offset:
      peptide_length = len(peptide_sequence)
      cyclic_offset_array = np.zeros((peptide_length, peptide_length))
      cyc_row = np.arange(0,-peptide_length,-1)
      pc = int(np.round(peptide_length/2)) #Get centre
      cyc_row[pc+1:]=np.arange(len(cyc_row[pc+1:]),0,-1)
      for i in range(len(cyclic_offset_array)):
          cyclic_offset_array[i]=np.roll(cyc_row,i)
      feature_dict['peptide_cyclic_offset_array']=cyclic_offset_array

  ####Run the directed evolution####
  sequence_scores = {'if_dist_peptide':[], 'if_dist_receptor':[],'plddt':[], 'delta_CM':[], 'loss':[],'sequence':[]}


  #Check if a run exists
  if os.path.exists(output_dir+'metrics.csv'):
      df = pd.read_csv(output_dir+'metrics.csv')
      for col in df.columns:
          sequence_scores[col] = [*df[col].values]
      #Peptide sequence
      peptide_sequence = sequence_scores['sequence'][np.argmin(sequence_scores['loss'])]




  if predict_only==True and predict_only_sequence:
      peptide_sequence = predict_only_sequence
      receptor_CAs, peptide_CM, receptor_if_residues = [],[],[]
  else:
      #Get the receptor CAs
      receptor_CAs = np.load(receptor_CAs)
      #Get the peptide centre of mass
      peptide_CM = np.load(peptide_CM)
      #Target residues
      receptor_if_residues = np.load(receptor_if_residues)


  if len(sequence_scores['if_dist_peptide'])<1:
      #Get an initial estimate
      if_dist_peptide, if_dist_receptor, plddt, delta_CM, unrelaxed_protein = predict_function(peptide_sequence, feature_dict, output_dir, model_runners,
                                                                                    random_seed, receptor_if_residues, receptor_CAs, peptide_CM, predict_only)

  #Save
  if predict_only==True:
      save_design(unrelaxed_protein, output_dir, 'true', feature_dict['seq_length'][0])
      sys.exit()
  else:
      if len(sequence_scores['if_dist_peptide'])<1:
          #Save
          sequence_scores['if_dist_peptide'].append(if_dist_peptide)
          sequence_scores['if_dist_receptor'].append(if_dist_receptor)
          sequence_scores['plddt'].append(plddt)
          sequence_scores['delta_CM'].append(delta_CM)
          if plDDT_only==True:
              loss = 1/plddt
          else:
              loss = (if_dist_peptide+if_dist_receptor)/2*1/plddt*delta_CM
          sequence_scores['loss'].append(loss)
          sequence_scores['sequence'].append(peptide_sequence)
          print(if_dist_peptide, if_dist_receptor, plddt, delta_CM, loss, peptide_sequence)

  #Iterate
  for num_iter in range(len(sequence_scores['if_dist_peptide'])-1, num_iterations):
    #Mutate sequence
    new_sequence = mutate_sequence(peptide_sequence, sequence_scores)
    #Predict and get loss
    if_dist_peptide, if_dist_receptor, plddt, delta_CM, unrelaxed_protein = predict_function(new_sequence, feature_dict, output_dir, model_runners,
                                                                    random_seed, receptor_if_residues, receptor_CAs, peptide_CM, predict_only)

    #Check if the loss improved
    if plDDT_only==True:
        loss = 1/plddt
    else:
        loss = (if_dist_peptide+if_dist_receptor)/2*1/plddt*delta_CM
    if loss<min(sequence_scores['loss']):
        #Create new best seq
        peptide_sequence = new_sequence

    #Save loss and weights
    sequence_scores['if_dist_peptide'].append(if_dist_peptide)
    sequence_scores['if_dist_receptor'].append(if_dist_receptor)
    sequence_scores['plddt'].append(plddt)
    sequence_scores['delta_CM'].append(delta_CM)
    sequence_scores['loss'].append(loss)
    sequence_scores['sequence'].append(new_sequence)

    print(num_iter, if_dist_peptide, if_dist_receptor, plddt, delta_CM, loss, peptide_sequence)

    #Save
    save_df = pd.DataFrame.from_dict(sequence_scores)
    save_df.to_csv(output_dir+'metrics.csv', index=None)

    save_design(unrelaxed_protein, output_dir, str(num_iter), feature_dict['seq_length'][0])

######################MAIN###########################
def main(argv):

  #Use a single ensemble
  num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_name = pathlib.Path(FLAGS.receptor_fasta_path).stem

  data_pipeline = foldonly.FoldDataPipeline()
  model_runners = {}
  for model_name in FLAGS.model_names:

    model_config = config.model_config(model_name)
    if FLAGS.cyclic_offset:
        model_config.model.embeddings_and_evoformer.cyclic_offset=True
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.common.num_recycle = FLAGS.max_recycles
    model_config.model.num_recycle = FLAGS.max_recycles
    model_params = data.get_model_haiku_params(
          model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
                 list(model_runners.keys()))
  amber_relaxer = None

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure for each of the sequences.
  optimise_binder(
        fasta_path=FLAGS.receptor_fasta_path,
        fasta_name=fasta_name,
        receptor_if_residues=FLAGS.receptor_if_residues,
        receptor_CAs=FLAGS.receptor_CAs,
        peptide_length=FLAGS.peptide_length,
        peptide_CM=FLAGS.peptide_CM,
        output_dir=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        random_seed=random_seed,
        num_iterations=FLAGS.num_iterations,
        predict_only=FLAGS.predict_only,
        predict_only_sequence=FLAGS.peptide_sequence,
        plDDT_only=FLAGS.plDDT_only)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'receptor_fasta_path',
      'receptor_if_residues',
      'peptide_length',
      'output_dir',
      'model_names',
      'data_dir'
  ])

  app.run(main)
