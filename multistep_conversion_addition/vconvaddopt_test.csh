#!/bin/csh
#$ -M name@email.com				# email address for job updates
#$ -m abe
#$ -pe smp 5					# request cores
#$ -q gpu
#$ -l gpu_card=1#$ -N vconvaddopt_test		# name for the job
#$ -t 5						# specify the number of jobs (defined in argslist below)
#$ -o vconvaddopt_test_results			# result file name

chmod +x ~/vconvaddopttest.py			# giving permissions to run the program

# specify the parameters for each job you want to run
set arglist=( "-o vconvaddopt_test_1 -m ../../results/workingdir_1/vconvaddopt_model_1 -t 5 -h1 270 -h2 270 -h3 0" \
	    "-o vconvaddopt_test_2 -m ../../results/workingdir_1/vconvaddopt_model_2 -t 5 -h1 270 -h2 270 -h3 0" \
	    "-o vconvaddopt_test_3 -m ../../results/workingdir_1/vconvaddopt_model_3 -t 5 -h1 270 -h2 270 -h3 0" \
	    "-o vconvaddopt_test_4 -m ../../results/workingdir_1/vconvaddopt_model_4 -t 5 -h1 270 -h2 270 -h3 0" \
	    "-o vconvaddopt_test_5 -m ../../results/workingdir_1/vconvaddopt_model_5 -t 5 -h1 270 -h2 270 -h3 0" \
                )

set args = "$arglist[$SGE_TASK_ID]"
set cmd = "../../vconvaddopttest.py ../../vconvadd_dataset50M.npz ../../vconvadd_dataset50M_test.npz $args"		# specify where the program and datasets are

module load python/3.6.0

set wd = "workingdir_${JOB_NAME}_${QUEUE}_${JOB_ID}_${SGE_TASK_ID}"
mkdir -p results/$wd
pushd results/$wd
$cmd >& vconvaddopt_test.log		# specify the log file


