
#############PARAMETERS#############
BASE=. #Change this depending on your local path
DATADIR=$BASE/data/test #The output (designs) will also be written here
RECEPTORID=1ssc_receptor
###Receptor interface residues - provide with --receptor_if_residues=$RECEPTORIFRES if using
RECEPTORIFRES="4,5,8,11,12,45,47,54,55,57,58,59,65,66,72,74,81,83,102,104,105,106,107,108,109,110,111,112"
####Receptor fasta sequence####
RECEPTORFASTA=$DATADIR/$RECEPTORID'.fasta'
###Peptide length###
PEPTIDELENGTH=10
#NUMBER OF ITERATIONS
NITER=1000 #This will likely have to be modified depending on the outcome of the design

#########Step1: Create MSA with HHblits#########
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
#Write individual fasta files for all unique sequences
if test -f $DATADIR/$RECEPTORID'.a3m'; then
	echo $DATADIR/$RECEPTORID'.a3m' exists
else
	$BASE/hh-suite/build/bin/hhblits -i $RECEPTORFASTA -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $DATADIR/$RECEPTORID'.a3m'
fi
#MSA
MSA=$DATADIR/$RECEPTORID'.a3m'


#########Step2: Design binder#########
##### AF2 CONFIGURATION #####
PARAM=$BASE'/src/AF2/'
PRESET='full_dbs' #Choose preset model configuration - no ensembling (full_dbs) and (reduced_dbs) or 8 model ensemblings (casp14).
MAX_RECYCLES=8 #max_recycles (default=3)
MODEL_NAME='model_1' #model_1_ptm
MSAS="$MSA" #Comma separated list of msa paths

#Optimise a binder
#conda activate evobind
python3 $BASE/src/mc_design.py \
--receptor_fasta_path=$RECEPTORFASTA \
--peptide_length=$PEPTIDELENGTH \
--msas=$MSAS \
--output_dir=$DATADIR/ \
--model_names=$MODEL_NAME \
--data_dir=$PARAM \
--max_recycles=$MAX_RECYCLES \
--num_iterations=$NITER \
--predict_only=False \
#--cyclic_offset=1
