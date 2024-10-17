AFM_DIR=../../data/test/AFM/
EVO_DIR=../../data/test/
PEP_SEQS=../../data/test/AFM/1ssc_metrics_1_10.csv
OUTDIR=../../data/test/AFM/

#Run
conda activate evobind
python3 ./afm_evo_loss_calc.py --afm_dir $AFM_DIR \
--evo_dir $EVO_DIR --score_csv $PEP_SEQS \
--outdir $OUTDIR
