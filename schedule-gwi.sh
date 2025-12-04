#!/bin/bash

###############################################################################
## Build the job index ########################################################

# Define the GWI method argv inputs. ##########################################

# Select the date to start the regression range from
START_REGRESS=1850

# Select the date to end the regression range at. Create a sequential range
# for the range of regressed years.
# This is for calculating the historical-only GWI:
# END_REGRESS=`seq 2000 2023`  # This is inclusive of the start and end years
# This is for calculating the GWI with all years:
END_REGRESS=`seq 2020 2024`

# Create array of subsampling sizes to calculate.
# This is for scaling up the calculation:
# SUBSAMPLE_ITERATIONS=(60 65 70 75 80 85 90 95 100)  # Size of subsampling
# This is for repeating final calculations at one size:
SUBSAMPLE_ITERATIONS=(20)  # Size of subsampling

# Select the reference period for the temperature datasets
# e.g. 1850-1900
# e.g. 1981-2010
PREINDUSTRIAL_ERA=1850-1900

# Select which variables to regress on.
# e.g. GHG,OHF,Nat
# e.g. Ant,Nat
# e.g. Tot
VARS=GHG,OHF,Nat

# Select whether to include sub-variables in the output
# i.e. this will calculate component-wise contributions to the aggregate
# variables (e.g. GHG = CO2 + CH4 + N2O + F-gases).
# e.g. y
# e.g. n
INCLUDE_SUB_VARS=y

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
# e.g. observed_JK-2024-SSP245
SCENARIO=observed-2024

# Select whether to consider committed warming at constant ERF.
# This amends/extends the scenario to hold ERF constant from a start year
# to an end year.
# Format: start_year-end_year
# You can use 'end_regress' as a keyword for the start year.
# e.g. n (no committed warming)
# e.g. 2024-2300 (constant ERF from 2024 to 2300)
# e.g. end_regress-2300 (constant ERF from the end of regression to 2300)
COMMITTED=n

# Select truncation range
TRUNCATION=1850-2024

# Select whether to include the rate of change in the regression
# e.g. y
# e.g. n
INCLUDE_RATE=n

# Select whether to include the headlines in the regression
# e.g. 'annual,SR1.5,AR6,CGWL'
# e.g. n
HEADLINE_TOGGLES='annual,AR6,SR1.5,CGWL'

# Select which years for the headlines to cover.
# If HEADLINES_TOGGLE is set to 'n' then this will be ignored.
# e.g. 'end_regress' for the end of the regressed range
# e.g. 'end_trunc' for the end of the truncation range
# e.g. 'IGCC' for latest year, 2017 repeat, and 2010-2019 repeat
# e.g. '2024' for a single year
# e.g. '2023,2024,2025' for multiple separate years.
# e.g. 'end_regress,2050,2100,2300' to combine end_regress and manual years
# e.g. $(seq -s, 1950 2024) will create a comma-separated list of years
HEADLINE_YEARS='end_regress'


# Select which ensemble members use from the scenario ERF/GMT files
# e.g. 'all'  # (Use all members for all sources).
# e.g. 1.  # (Use single member only)
# e.g. {0..49}  # (Use single member only, and apply separately to each member
# in the range)
SPECIFY_ENSEMBLE_MEMBERS=all


# Select which uncertainty sources this single-member selection should apply
# to.
# NOTE: For now, we just consider ERF and GMT specification.
# By default, all ensemble members will be used for an uncertainty source;
# If SPECIFY_ENSEMBLE_MEMBERS is not set to 'all', then only the specified
# members will be used for the sources listed here; sources not listed here
# will use all ensemble members.
# e.g. 'ERF'  # (Apply above selection to ERF only)
# e.g. 'GMT'  # (Apply above selection to GMT only)
# e.g. 'ERF,GMT'  # (Apply above selection to both ERF and GMT; in this case,
# the same ensemble members will be paired for both sources;
# i.e. ensemble member 0 for ERF is paired with ensemble member 0 for GMT)
SPECIFY_ENSEMBLE_MEMBER_SOURCE_FOR='GMT'


###############################################################################
### Generate a Slurm file for each Job ID #####################################

WALLTIME=2:00:00
PARTITION=Short
SIM_NAME=gwi
SIM_CPUS=28
SLURM_FILE_NAME=${SIM_NAME}_${START_REGRESS}-
LOG_DIR=slurm_logs
mkdir -p ${LOG_DIR}

# Keep track of which iteration we are on (avoid overwriting log files)
count=1
# Create the job file for each job ID
for j in "${SUBSAMPLE_ITERATIONS[@]}"
do

for i in $END_REGRESS
# for i in "${END_REGRESS[@]}"
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
#SBATCH --mem-per-cpu=8192
#SBATCH --partition=${PARTITION}

## Name the job and queue it
#SBATCH --job-name=${SIM_NAME}_${SCENARIO}_${START_REGRESS}-${i}_${j}_${count}

## Declare an output log for all jobs to use:
#SBATCH --output=./${LOG_DIR}/${SIM_NAME}_${SCENARIO}_${VARS}_${START_REGRESS}-${i}_${j}_${count}.out

# For the ARC cluster
# module load Mamba
# module load Miniconda3
# conda activate gwi-new
# micromamba activate gwi-mamba

# For the single ensemble member selection runs
if [[ "${SPECIFY_ENSEMBLE_MEMBERS}" == "all" ]]; then
  # Regress against all reference temperatures at the same time
  python gwi.py --samples=${j} --regress-range=${START_REGRESS}-${i} --truncate=${TRUNCATION} --include-rate=${INCLUDE_RATE} --headline-toggles=${HEADLINE_TOGGLES} --headline-years=${HEADLINE_YEARS}  --regress-variables=${VARS} --scenario=${SCENARIO} --committed=${COMMITTED} --preindustrial-era=${PREINDUSTRIAL_ERA} --specify-ensemble-member-sources-for=${SPECIFY_ENSEMBLE_MEMBER_SOURCE_FOR} --specify-ensemble-member=${SPECIFY_ENSEMBLE_MEMBERS} --include-sub-vars=${INCLUDE_SUB_VARS}
else
  for k in ${SPECIFY_ENSEMBLE_MEMBERS}; do
    # Regress against each reference temperature separately
    python gwi.py --samples=${j} --regress-range=${START_REGRESS}-${i} --truncate=${TRUNCATION} --include-rate=${INCLUDE_RATE} --headline-toggles=${HEADLINE_TOGGLES} --headline-years=${HEADLINE_YEARS}  --regress-variables=${VARS} --scenario=${SCENARIO} --committed=${COMMITTED} --preindustrial-era=${PREINDUSTRIAL_ERA} --specify-ensemble-member-sources-for=${SPECIFY_ENSEMBLE_MEMBER_SOURCE_FOR} --specify-ensemble-member=\$k --include-sub-vars=${INCLUDE_SUB_VARS}
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
# size. i.e. for each member of SUBSAMPLE_ITERATIONS, this counter will increment.
count=$((count + 1))

done
