

#############PARAMETERS#############
# Figure out dir of the script, so we can launch from anywhere
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) #From Toni Giorgino
BASE=$SCRIPT_DIR
DATADIR=$BASE/data/test
RECEPTORID=1ssc_receptor
###Receptor interface residues
RECEPTORIFRES=$DATADIR/$RECEPTORID'_target_residues.npy'
#Receptor CAs
RECEPTOR_CAS=$DATADIR/$RECEPTORID'_CA.npy'
####Receptor fasta sequence####
RECEPTORFASTA=$DATADIR/$RECEPTORID'.fasta'
###Peptide length###
PEPTIDELENGTH=11
###Peptide centre of mass###
PEPTIDE_CM=$DATADIR/1ssc_CM.npy
#NUMBER OF ITERATIONS
NITER=300 #This will likely have to be modified depending on the outcome of the design
#Path to singularity - now this is hardcoded assuming the path created from ./src/install_singularity_ubuntu.sh, if you have singularity in your path change this variable
SINGULARITY=/opt/singularity3/bin/singularity

#########Step1: Create MSA with HHblits#########
SINGIMG=$BASE/src/AF2/AF_environment.sif #Sing img
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
#Write individual fasta files for all unique sequences
if test -f $DATADIR/$RECEPTORID'.a3m'; then
	echo $DATADIR/$RECEPTORID'.a3m' exists
else
	$SINGULARITY exec $SINGIMG hhblits -i $RECEPTORFASTA -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $DATADIR/$RECEPTORID'.a3m'
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
$SINGULARITY exec --nv --bind $BASE:$BASE $SINGIMG \
python3 $BASE/src/mc_design.py \
		--receptor_fasta_path=$RECEPTORFASTA \
		--receptor_if_residues=$RECEPTORIFRES \
		--receptor_CAs=$RECEPTOR_CAS \
		--peptide_length=$PEPTIDELENGTH \
		--peptide_CM=$PEPTIDE_CM \
		--msas=$MSAS \
		--output_dir=$DATADIR \
		--model_names=$MODEL_NAME \
		--data_dir=$PARAM \
		--max_recycles=$MAX_RECYCLES \
		--num_iterations=$NITER \
		--predict_only=False \
