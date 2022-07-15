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

"""Full AlphaFold protein structure prediction script."""
import json
import os
import warnings
import pathlib
import pickle
import random
import sys
import time
from typing import Dict, Optional

from absl import app
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
# Internal import (7716).


MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline,
    benchmark: bool,
    random_seed: int,
    model_runners: Optional[Dict[str, model.RunModel]],
    msas: list,
    chain_break_list: list):

  """Predicts structure using AlphaFold for the given sequence."""
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Get features.
  t_0 = time.time()
  feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path,
        input_msas=msas,
        template_search=None,
        msa_output_dir=msa_output_dir)
  timings['features'] = time.time() - t_0

  # Introduce chain breaks for oligomers ########## NEW!
  idx_res = feature_dict['residue_index']
  prev_overlay = 0

  for chain_break in chain_break_list:
    try: chain_break = int(chain_break.strip())
    except: raise TypeError('--chain_break_list argument must be comma separated list'
                              'of lengths of each concatenated chain in the order they '
                              'appear in the input fasta.')
    if chain_break not in list(range(len(idx_res))):
      if chain_break == chain_break_list[-1]: break
      else: raise ValueError('Specified chain break {} does not appear in sequence length of {}.'\
                               .format(chain_break, len(idx_res)))
    idx_res[chain_break:] += 200
  feature_dict['residue_index'] = idx_res

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  with open(features_output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

  relaxed_pdbs = {}
  plddts = {}

  # Run the models.
  for model_name, model_runner in model_runners.items():
    logging.info('Running model %s', model_name)
    t_0 = time.time()
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=random_seed)
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(processed_feature_dict)
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s predict time (includes compilation time, see --benchmark): %.0f?',
        model_name, t_diff)

    if benchmark:
      t_0 = time.time()
      model_runner.predict(processed_feature_dict)
      timings[f'predict_benchmark_{model_name}'] = time.time() - t_0

    # Get mean pLDDT confidence metric.
    plddt = prediction_result['plddt']
    plddts[model_name] = np.mean(plddt)

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors)

    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(protein.to_pdb(unrelaxed_protein))

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(model_names, num_ensemble, max_recycles, data_dir, fasta_path, fasta_name, msas, chain_break_list, output_dir):
  model_runners = {}
  for model_name in model_names:
    model_config = config.model_config(model_name)
    model_config.data.eval.num_ensemble = num_ensemble
    model_config.data.common.num_recycle = max_recycles
    model_config.model.num_recycle = max_recycles
    model_params = data.get_model_haiku_params(
      model_name=model_name, data_dir=data_dir)
    model_runner = model.RunModel(model_config, model_params)
    model_runners[model_name] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
             list(model_runners.keys()))

  random_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed %d for the data pipeline', random_seed)

  data_pipeline = foldonly.FoldDataPipeline()
  # Predict structure
  predict_structure(fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        benchmark=False,
        random_seed=random_seed,
        msas=msas,
        chain_break_list=chain_break_list)
