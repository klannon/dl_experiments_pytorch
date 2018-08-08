#!/bin/csh
#$ -M example@email.com                	     # email address for job updates
#$ -m abe
#$ -pe smp 5			             # request cores
#$ -q gpu
#$ -l gpu_card=1#$ -N vconv2hl_single        # name for the job
#$ -t 1-3			             # specify the number of jobs (defined in argslist below)
#$ -o vconv2hl_singleresults	 	     # result file name

chmod +x ~/vconv2hl_single.py	             # giving permissions to run the program

# specify the parameters for each job you want to run
set arglist=( "-o vconv2hl_1 -h 500 -l 500 -i 10 -t 5 -b 250 -n 400 -v 30000" \
        "-o vconv2hl_2 -h 250 -l 250 -i 10 -t 5 -b 250 -n 400 -v 30000" \
        "-o vconv2hl_3 -h 1000 -l 1000 -i 10 -t 5 -b 250 -n 400 -v 30000" \
		)

set args = "$arglist[$SGE_TASK_ID]"
set cmd = "../../vconv2hl_single.py ../../vconvdataset10M.npz ../../vconvdataset10M_test.npz $args"	# specify where the program and datasets are

module load python/3.6.0

set wd = "workingdir_${JOB_NAME}_${QUEUE}_${JOB_ID}_${SGE_TASK_ID}"
mkdir -p results/$wd
pushd results/$wd
$cmd >& vconv2hl_single.log       	# specify the log file
