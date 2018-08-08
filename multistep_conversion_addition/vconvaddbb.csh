#!/bin/csh
#$ -M name@email.com			# email address for job updates
#$ -m abe
#$ -pe smp 3				# request cores
#$ -q gpu
#$ -l gpu_card=1#$ -N vconvaddbb	# name for the job
#$ -t 1-3 				# specify the number of jobs (defined in argslist below)
#$ -o vconvaddbb_results		# result file name

chmod +x ~/vconvaddbb.py		# giving permissions to run the program

# specify the parameters for each job you want to run
set arglist=("-o vconvaddbb_1 -l1 540 -l2 540 -b 250 -n 500 -v 10000000 -t 5" \
	     "-o vconvaddbb_2 -l1 270 -l2 270 -b 250 -n 500 -v 10000000 -t 5" \
	     "-o vconvaddbb_3 -l1 135 -l2 135 -b 250 -n 500 -v 10000000 -t 5" \
                )

set args = "$arglist[$SGE_TASK_ID]"
set cmd = "../../vconvaddbb.py ../../vconvadd_dataset50M.npz ../../vconvadd_dataset50M_test.npz $args"		# specify where the program and datasets are


module load python/3.6.0

set wd = "workingdir_${JOB_NAME}_${QUEUE}_${JOB_ID}_${SGE_TASK_ID}"
mkdir -p results/$wd
pushd results/$wd
$cmd >& vconvaddbb.log		 # specify the log file

