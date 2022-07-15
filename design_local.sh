

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
####Get Receptor MSA####

####Create MSA with HHblits####
SINGIMG=$BASE/src/AF2/AF_environment.sif #Sing img
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
#Write individual fasta files for all unique sequences
hhblits -i $RECEPTORFASTA -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $DATADIR/$RECEPTORID'.a3m'
#MSA
MSA=$DATADIR/$RECEPTORID'.a3m'

##### AF2 CONFIGURATION #####
AFHOME='./'

#Singularity image
IMG=/home/pbryant/singularity_ims/af_torch_sbox

### path of param folder containing AF2 Neural Net parameters.
### download from: https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar)
PARAM=/home/pbryant/data/af_params/
### Path where AF2 generates its output folder structure
OUTFOLDER='../data/test/'

### Running options for obtaining a refines tructure ###
MAX_RECYCLES=8 #max_recycles (default=3)
MODEL_NAME='model_1' #model_1_ptm
MSAS="$MSA" #Comma separated list of msa paths

#NUMBER OF ITERATIONS
NITER=1000

#Regarding the run mode
SINGULARITY=/opt/singularity3/bin/singularity
$SINGULARITY exec --nv $IMG \
python3 $AFHOME/mc_design.py \
		--receptor_fasta_path=$RECEPTORFASTA \
		--receptor_if_residues=$RECEPTORIFRES \
		--receptor_CAs=$RECEPTOR_CAS \
		--peptide_length=$PEPTIDELENGTH \
		--peptide_CM=$PEPTIDE_CM \
		--msas=$MSAS \
		--output_dir=$OUTFOLDER \
		--model_names=$MODEL_NAME \
	  --data_dir=$PARAM \
		--max_recycles=$MAX_RECYCLES \
		--num_iterations=$NITER \
		--predict_only=False \
