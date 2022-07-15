

#############PARAMETERS#############
BASE=$(pwd) #Where all scripts are run from, now the current directory
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
NITER=300


#########Step1: Create MSA with HHblits#########
SINGIMG=$BASE/src/AF2/AF_environment.sif #Sing img
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
#Write individual fasta files for all unique sequences
if test -f $DATADIR/$RECEPTORID'.a3m'; then
	echo $DATADIR/$RECEPTORID'.a3m' exists
else
	hhblits -i $RECEPTORFASTA -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $DATADIR/$RECEPTORID'.a3m'
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
SINGULARITY=/opt/singularity3/bin/singularity
$SINGULARITY exec --nv $IMG \
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
