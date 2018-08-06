#!/bin/csh
#$ -M kklanot@nd.edu
#$ -m abe
#$ -pe smp 1
#$ -q gpu
#$ -l gpu_card=1#$ -N vconvaddopt_train
#$ -t 1
#$ -o vconvaddopt_train

chmod +X ~/vconvaddopttrain.py

set arglist=("-o vconvaddopt_train_1 -b 50 -n 400 -v 10000000" \
                )

set args = "$arglist[$SGE_TASK_ID]"
set cmd = "../../vconvaddopttrain.py ../../vconvadd_dataset10M.npz $args"

module load python/3.6.0

setenv CUDA_VISIBLE_DEVICES `~/find_cvd.sh`

set wd = "workingdir_${JOB_NAME}_${QUEUE}_${JOB_ID}_${SGE_TASK_ID}"
mkdir -p results/$wd
pushd results/$wd
$cmd >& vconvaddopt_train.log

