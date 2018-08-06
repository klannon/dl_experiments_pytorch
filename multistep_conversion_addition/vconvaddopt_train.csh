#!/bin/csh
#$ -M name@email.com                          # email address for job updates
#$ -m abe
#$ -pe smp 4                                  # request cores
#$ -q gpu
#$ -l gpu_card=1#$ -N vconvaddopt_train       # name for the job
#$ -t 4                                       # specify the number of jobs (defined in argslist below)
#$ -o vconvaddopt_train                       # result file name

chmod +X ~/vconvaddopttrain.py                # giving permissions to run the program

# specify the parameters for each job you want to run
set arglist=("-o vconvaddopt_model_1 -b 250 -n 500 -v 10000000 -t 5 -h1 270 -h2 0 -h3 0" \
             "-o vconvaddopt_model_2 -b 250 -n 500 -v 10000000 -t 5 -h1 270 -h2 270 -h3 0" \
             "-o vconvaddopt_model_3 -b 250 -n 500 -v 10000000 -t 5 -h1 135 -h2 135 -h3 0" \
             "-o vconvaddopt_model_4 -b 250 -n 500 -v 10000000 -t 5 -h1 100 -h2 100 -h3 70" \
                )

set args = "$arglist[$SGE_TASK_ID]"
set cmd = "../../vconvaddopttrain.py ../../vconvadd_dataset50M.npz $args"     # specify where the program and datasets are

module load python/3.6.0

set wd = "workingdir_${JOB_NAME}_${QUEUE}_${JOB_ID}_${SGE_TASK_ID}"
mkdir -p results/$wd
pushd results/$wd
$cmd >& vconvaddopt_train.log            # specify the log file

