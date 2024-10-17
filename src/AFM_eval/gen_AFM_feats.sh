"""
Script for generating input features with AlphaFold-multimer

Fill in all the variables below to run the feature generation.
The reduced database version is used here.

This script assumes that all python packages necessary are in the current path.
"""

#Get ID
ID=1ssc
echo $ID
#Generate input MSAs and templates for AFM
FASTA_PATHS=../../data/test/AFM/1ssc.fasta
ls $FASTA_PATHS
OUTDIR=../../data/test/AFM/
#Genetic search
BASE=./ #Path to where you have all your software for AFM installed
JACKHMMER=$BASE/hmmer-3.4/src/jackhmmer
HHBLITS=$BASE/hh-suite/build/bin/hhblits
HHSEARCH=$BASE/hh-suite/build/bin/hhsearch
HMMSEARCH=$BASE/hmmer-3.4/src/hmmsearch
HMMBUILD=$BASE/hmmer-3.4/src/hmmbuild
KALIGN=$BASE/kalign-3.4/src/kalign
#Dbs
DATADIR=./ #Path to where you have all the AFM databases
UNIREF90=$DATADIR/uniref90/uniref90.fasta
MGNIFY=$DATADIR/mgnify/mgy_clusters_2022_05.fa
#BFD=$DATADIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
SMALL_BFD=$DATADIR/small_bfd/bfd-first_non_consensus_sequences.fasta
UNIREF30=$DATADIR/uniref30/UniRef30_2021_03
UNIPROT=$DATADIR/uniprot/uniprot.fasta
PDB70=$DATADIR/pdb70_from_mmcif_220313
PDBSEQRES=$DATADIR/pdb_seqres/pdb_seqres.txt
MMCIFDIR=$DATADIR/pdb_mmcif/mmcif_files/
#Settings
DB_PRESET='reduced_dbs'
MODEL_PRESET='multimer'
MAX_DATE='2034-01-01' #Max template date
OBS_PDBS=../../data/test/AFM/obsolete.txt
echo "" > $OBS_PDBS
AFDIR=./
#Run
python3 $AFDIR/run_alphafold_msa_template_only.py --fasta_paths=$FASTA_PATHS \
	--output_dir=$OUTDIR --jackhmmer_binary_path=$JACKHMMER \
	--hhblits_binary_path=$HHBLITS --hhsearch_binary_path=$HHSEARCH \
	--hmmsearch_binary_path=$HMMSEARCH --hmmbuild_binary_path=$HMMBUILD \
	--kalign_binary_path=$KALIGN --uniref90_database_path=$UNIREF90 \
	--mgnify_database_path=$MGNIFY --small_bfd_database_path=$SMALL_BFD \
	--uniprot_database_path=$UNIPROT \
	--pdb_seqres_database_path=$PDBSEQRES \
	--template_mmcif_dir=$MMCIFDIR --db_preset=$DB_PRESET --model_preset=$MODEL_PRESET \
	--max_template_date=$MAX_DATE --obsolete_pdbs_path=$OBS_PDBS
