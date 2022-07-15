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

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.templates import TemplateSearchResult
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer
import numpy as np

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]

TEMPLATE_FEATURES = {
      'template_aatype': np.float32,
      'template_all_atom_masks': np.float32,
      'template_all_atom_positions': np.float32,
      'template_domain_names': np.object,
      'template_sequence': np.object,
      'template_sum_probs': np.float32}

def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features


class FoldDataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self, pdb70_database_path: str = None,
               template_featurizer: templates.TemplateHitFeaturizer = None):
    if pdb70_database_path and template_featurizer:
      self.template_featurizer = template_featurizer

  def process(self, input_fasta_path: str, input_msas: list, 
              msa_output_dir: str, template_search: Optional[str]) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_desc = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_desc[0]
    num_res = len(input_sequence)

    parsed_msas = []
    parsed_delmat = []
    for custom_msa in input_msas:
      msa = ''.join([line for line in open(custom_msa)])
      if custom_msa[-3:] == 'sto': 
        parsed_msa, parsed_deletion_matrix, _ = parsers.parse_stockholm(msa) 
      elif custom_msa[-3:] == 'a3m':
        parsed_msa, parsed_deletion_matrix = parsers.parse_a3m(msa)
      else: raise TypeError('Unknown format for input MSA, please make sure '
                            'the MSA files you provide terminates with (and '
                            'are formatted as) .sto or .a3m')
      parsed_msas.append(parsed_msa)
      parsed_delmat.append(parsed_deletion_matrix)

    if template_search:
      hhsearch_result = ''.join([line for line in template_search])
      hhsearch_hits = parsers.parse_hhr(hhsearch_result)
      templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        query_pdb_code=None,
        query_release_date=None,
        hits=hhsearch_hits)
    else:
      template_features = {}
      for name in TEMPLATE_FEATURES:
        template_features[name] = np.array([], dtype=TEMPLATE_FEATURES[name])
        templates_result = TemplateSearchResult(features=template_features, errors=[], warnings=[])

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features(
        msas=parsed_msas, deletion_matrices=parsed_delmat)

    for n, msa in enumerate(parsed_msas):
        logging.info('MSA %d size: %d sequences.', n, len(msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features, **msa_features, **templates_result.features}
