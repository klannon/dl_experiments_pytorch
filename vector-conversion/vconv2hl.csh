#!/bin/csh
#$ -M kklanot@nd.edu
#$ -m abe
#$ -pe smp 5
#$ -q gpu
#$ -l gpu_card=1#$ -N vconv1hl
#$ -t 1-4
#$ -o vconv2hlresults

chmod +x ~/vconvtrain2hl.py

set arglist=( "-o vconv2hl_1 -l 500 -s 10 -e 100 -i 10 -t 3 -b 250 -n 400 -v 30000" \
        "-o vconv2hl_2 -l 500 -s 110 -e 200 -i 10 -t 3 -b 250 -n 400 -v 30000" \
        "-o vconv2hl_3 -l 500 -s 210 -e 300 -i 10 -t 3 -b 250 -n 400 -v 30000" \
        "-o vconv2hl_3 -l 500 -s 310 -e 400 -i 10 -t 3 -b 250 -n 400 -v 30000" \
        "-o vconv2hl_4 -l 500 -s 410 -e 500 -i 10 -t 3 -b 250 -n 400 -v 30000" \
		)

set args = "$arglist[$SGE_TASK_ID]"
set cmd = "../../vconvtrain.py ../../vconvdataset.npz ../../vconvdataset_test.npz $args"

module load python/3.6.0

set wd = "workingdir_${JOB_NAME}_${QUEUE}_${JOB_ID}_${SGE_TASK_ID}"
mkdir -p results/$wd
pushd results/$wd

