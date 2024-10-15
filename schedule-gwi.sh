#!/bin/bash

## Build the job index ########################################################

# Create a sequential range for the range of regressed years.
# This is for calculating the historical-only GWI:
# array_values=`seq 2000 2023`  # This is inclusive of the start and end years
# This is for calculating the GWI with all years:
array_values=`seq 1990 2023`

# Create array of subsampling sizes to calculate.
# This is for scaling up the calculation:
# array_samples=(60 65 70 75 80 85 90 95 100)  # Size of subsampling
# This is for repeating final calculations at one size:
array_samples=(60 60 60)  # Size of subsampling

# Select which variables to regress on.
# e.g. VARS=GHG,OHF,Nat
# e.g. VARS=Ant,Nat
# e.g. VARS=Tot
VARS=Tot

### Generate a Slurm file for each Job ID #####################################

WALLTIME=2:00:00
SIM_NAME=gwi-hist
SIM_CPUS=28
SLURM_FILE_NAME=${SIM_NAME}_1850-
LOG_DIR=slurm_logs
mkdir -p ${LOG_DIR}

# Keep track of which iteration we are on (avoid overwriting log files)
count=1
# Create the job file for each job ID
for j in "${array_samples[@]}"
do

for i in $array_values
do
echo $count
cat > ${SLURM_FILE_NAME}${i}_${j}_${VARS}_${count}.slurm << EOF
#!/bin/bash
#
## Set the maximum amount of runtime
#SBATCH --time=${WALLTIME}

## Request one node with many cpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${SIM_CPUS}
#SBATCH --mem-per-cpu=8192

## Name the job and queue it
#SBATCH --job-name=${SIM_NAME}_1850-${i}_${j}_${count}

## Declare an output log for all jobs to use:
#SBATCH --output=./${LOG_DIR}/${SIM_NAME}_${VARS}_1850-${i}_${j}_${count}.out

python gwi.py --samples=${j} --regress-range=1850-${i} --include-rate=n --include-headlines=n --regress-variables=${VARS}
EOF

# Submit a single job to slurm.
sbatch ${SLURM_FILE_NAME}${i}_${j}_${VARS}_${count}.slurm

# Remove the job file as slurm reads the script at submission time and it is
# no longer needed.
rm -rf ${SLURM_FILE_NAME}${i}_${j}_${VARS}_${count}.slurm

done

# Increment the counter that keeps track of multiple runs at the same sample
# size. i.e. for each member of array_samples, this counter will increment.
count=$((count + 1))

done
