REC_FEATS=../../data/test/AFM/1ssc_receptor_feats.pkl
PEP_SEQS=../../data/test/AFM/1ssc_metrics_1_10.csv #Replace this with your own scores
PARAMDIR=./data/
MODEL_PRESET='multimer'
NUM_RECYCLES=20 #Number of recycles
CYCLIC=0 #Set to 1 if cyclic
OUTDIR=../../data/test/AFM/

#Run
conda activate evobind
python3 ./run_AFM.py --receptor_features $REC_FEATS \
--data_dir $PARAMDIR --model_preset $MODEL_PRESET \
--peptide_sequences $PEP_SEQS \
--num_recycles $NUM_RECYCLES \
--cyclic_offset $CYCLIC \
--outdir $OUTDIR
