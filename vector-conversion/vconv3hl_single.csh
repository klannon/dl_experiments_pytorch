#!/bin/csh
#$ -M example@email.com              		# email address for job updates
#$ -m abe
#$ -pe smp 3					# request cores
#$ -q gpu
#$ -l gpu_card=1#$ -N vconv3hl_single		# name for the job	
#$ -t 1-3
#$ -o vconv3hl_singleresults			# result file name

chmod +x ~/vconv3hl_single.py		 	# giving permissions to run the program

# specify the parameters for each job you want to run
set arglist=( "-o vconv3hl_1 -l1 10 -l2 9 -t 5 -b 250 -n 500 -v 20000" \
        "-o vconv3hl_2 -l1 30 -l2 3 -t 5 -b 250 -n 500 -v 20000" \
        "-o vconv3hl_3 -l1 18 -l2 5 -t 5 -b 250 -n 500 -v 20000" \
        	)

set args = "$arglist[$SGE_TASK_ID]"
set cmd = "../../vconv3hl_single.py ../../vconvdataset.npz ../../vconvdataset_test.npz $args"		 # specify where the program and datasets are

module load python/3.6.0

set wd = "workingdir_${JOB_NAME}_${QUEUE}_${JOB_ID}_${SGE_TASK_ID}"
mkdir -p results/$wd
pushd results/$wd
$cmd >& vconv3hl_single.log		# specify the log file

