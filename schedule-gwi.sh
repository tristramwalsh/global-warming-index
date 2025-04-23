#!/bin/bash

###############################################################################
## Build the job index ########################################################

# Define the GWI method argv inputs. ##########################################

# Select the date to start the regression range from
START_REGRESS=1850

# Select the date to end the regression range at. Create a sequential range
# for the range of regressed years.
# This is for calculating the historical-only GWI:
# array_values=`seq 2000 2023`  # This is inclusive of the start and end years
# This is for calculating the GWI with all years:
array_values=`seq 2020 2023`

# Create array of subsampling sizes to calculate.
# This is for scaling up the calculation:
# array_samples=(60 65 70 75 80 85 90 95 100)  # Size of subsampling
# This is for repeating final calculations at one size:
array_samples=(90 90 90)  # Size of subsampling

# Select the reference period for the temperature datasets
# e.g. 1850-1900
# e.g. 1981-2010
PREINDUSTRIAL_ERA=1850-1900

# Select which variables to regress on.
# e.g. GHG,OHF,Nat
# e.g. Ant,Nat
# e.g. Tot
VARS=GHG,OHF,Nat

# Select which scenario to analyse
# e.g. observed
# e.g. SMILE_ESM-SSP370
# e.g. SMILE_ESM-SSP245
# e.g. SMILE_ESM-SSP126
# e.g. observed-2023
# e.g. observed-2024
# e.g. observed-SSP119
# e.g. NorESM_rcp45-Volc
# e.g. NorESM_rcp45-VolcConst
SCENARIO=observed-2024

# Select truncation range
TRUNCATION=1850-2024

# Select whether to include the rate of change in the regression
# e.g. y
# e.g. n
INCLUDE_RATE=n

# Select whether to include the headlines in the regression
# e.g. y
# e.g. n
INCLUDE_HEADLINES=y

# Select which ensemble members use from the scenario ERF/Temp files
# e.g. all
# e.g. 1
# e.g. {0..49}
SPECIFY_ENSEMBLE_MEMBERS=all
# SPECIFY_ENSEMBLE_MEMBERS={1..60}


###############################################################################
### Generate a Slurm file for each Job ID #####################################

WALLTIME=2:00:00
SIM_NAME=gwi
SIM_CPUS=28
SLURM_FILE_NAME=${SIM_NAME}_${START_REGRESS}-
LOG_DIR=slurm_logs
mkdir -p ${LOG_DIR}

# Keep track of which iteration we are on (avoid overwriting log files)
count=1
# Create the job file for each job ID
for j in "${array_samples[@]}"
do

for i in $array_values
# for i in "${array_values[@]}"
do
# 

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
##SBATCH --mem-per-cpu=8192
#SBATCH --mem=300G
#SBATCH --partition=short

## Name the job and queue it
#SBATCH --job-name=${SIM_NAME}_${SCENARIO}_${START_REGRESS}-${i}_${j}_${count}

## Declare an output log for all jobs to use:
#SBATCH --output=./${LOG_DIR}/${SIM_NAME}_${SCENARIO}_${VARS}_${START_REGRESS}-${i}_${j}_${count}.out

# For the ARC cluster
# module load Mamba
# module load Miniconda3
# conda activate gwi-new

# For the single ensemble member selection runs
if [[ "${SPECIFY_ENSEMBLE_MEMBERS}" == "all" ]]; then
  # Regress against all reference temperatures at the same time
  python gwi.py --samples=${j} --regress-range=${START_REGRESS}-${i} --truncate=${TRUNCATION} --include-rate=${INCLUDE_RATE} --include-headlines=${INCLUDE_HEADLINES} --regress-variables=${VARS} --scenario=${SCENARIO} --preindustrial-era=${PREINDUSTRIAL_ERA} --specify-ensemble-member=${SPECIFY_ENSEMBLE_MEMBERS}
else
  for k in ${SPECIFY_ENSEMBLE_MEMBERS}; do
    # Regress against each reference temperature separately
    python gwi.py --samples=${j} --regress-range=${START_REGRESS}-${i} --truncate=${TRUNCATION} --include-rate=${INCLUDE_RATE} --include-headlines=${INCLUDE_HEADLINES} --regress-variables=${VARS} --scenario=${SCENARIO} --preindustrial-era=${PREINDUSTRIAL_ERA} --specify-ensemble-member=\$k
  done
fi


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
