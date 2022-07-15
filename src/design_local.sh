
RECEPTORID=1ssc_receptor

####Get fasta file####
FASTADIR=./data/test
RECEPTORFASTA=$FASTADIR/$RECEPTORID'.fasta'

####Create MSA with HHblits####

###Peptide length###
PEPTIDELENGTH=11
###Peptide centre of mass###
PEPTIDE_CM=./data/test/1ssc_CM.npy
####Get Receptor MSA####
#HHblits
MSADIR=./data/test
MSA=$MSADIR/$RECEPTORID'.a3m'

###Get receptor interface residues
RECEPTORDIR=../data/PDB/receptor
RECEPTORIFRES=$RECEPTORDIR/$RECEPTORID'_if.npy'
#Receptor CAs
RECEPTOR_CAS=../data/PDB/1fiw_receptor_CA.npy
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
