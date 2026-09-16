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
END_REGRESS=`seq 1950 2025`

# Create array of subsampling sizes to calculate.
# This is for scaling up the calculation:
# SUBSAMPLE_ITERATIONS=(60 65 70 75 80 85 90 95 100)  # Size of subsampling
# This is for repeating final calculations at one size:
SUBSAMPLE_ITERATIONS=(60 60 60)  # Size of subsampling

# Select the reference period for the temperature datasets
# The selected period offset applies to FaIR outputs, GMT Observations,
# and piControl internal variability.
# Note that 'n' can be used to disable this preprocessing offset.
# e.g. 1850-1900
# e.g. 1981-2010
PREINDUSTRIAL_ERA=1850-1900

# Select whether to include a constant term offset in the multi-variable regression
# e.g. y (include the constant)
# e.g. n (do not include the constant term)
INCLUDE_REG_CONST=n

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
INCLUDE_SUB_VARS=n

# Select which scenario to analyse
# e.g. observed
# e.g. SMILE_ESM-SSP370
# e.g. SMILE_ESM-SSP245
# e.g. SMILE_ESM-SSP126
# e.g. observed-2023
# e.g. observed-2024
# e.g. observed-2025
# e.g. observed-2025-SSP119  (2025 observations, SSP119-extended ERFs)
# e.g. observed-2024-SSP245  (2024 observations, SSP245-extended ERFs)
# e.g. NorESM_rcp45-Volc
# e.g. NorESM_rcp45-VolcConst
# e.g. observed_JK-2024-SSP245
SCENARIO=observed-2025

# Select whether to consider committed warming at constant ERF.
# This amends/extends the scenario to hold ERF constant from a start year
# to an end year.
# Format: start_year-end_year
# You can use 'end_regress' as a keyword for the start year.
# e.g. n (no committed warming)
# e.g. 2024-2300 (constant ERF from 2024 to 2300)
# e.g. end_regress-2300 (constant ERF from the end of regression to 2300)
COMMITTED=2025-2050

# Select truncation range
TRUNCATION=1850-2025

# Select whether to include the rate of change in the regression
#TODO: Specify which years to include rate of change for.
# e.g. y
# e.g. n
INCLUDE_RATE=n

# Select whether to calculate the prior (pre-constrained) warming output.
# The priors depend only on the ERF ensemble and FaIR parameters, NOT on the
# reference temperature member. These priors are not used directly in the GWI
# calculation, but are useful for understanding the prior distribution of
# warming pre-constraint. Setting this to False is useful for the single-member
# selection runs, where the priors are identical for all members and need only
# be calculated once.
# e.g. y
# e.g. n
CALCULATE_PRIORS_OUTPUT=n

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
# NOTE: member labels are the dataset's own column names, and are 1-indexed:
# HadCRUT ships 200 realisations ({1..200}), the John Kennedy ensemble 100
# ({1..100}).
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
SPECIFY_ENSEMBLE_MEMBER_SOURCE_FOR='GMT,ERF'


###############################################################################
### Generate a Slurm file for each Job ID #####################################

# Select which Slurm cluster to submit to.
# Oxford ARC runs two clusters under one Slurm accounting database: 'arc'
# (large nodes, 48+ cores) and 'htc' (smaller nodes, GPUs). List them with
# `sacctmgr show cluster` or `sinfo -M all`.
# Jobs are assigned to a cluster once, at submission time, and never migrate
# afterwards; without --clusters, sbatch submits to whichever cluster you
# happen to be logged into.
# e.g. arc      (force ARC)
# e.g. htc      (force HTC)
# e.g. arc,htc  (let sbatch pick whichever offers the earliest start time)
CLUSTER=htc

if hostname | grep -Eq "htc|arc"; then  # ARC cluster
  PARTITION=short
  CLUSTER_DIRECTIVE="#SBATCH --clusters=${CLUSTER}"
elif hostname | grep -q "ouce"; then  # OUCE cluster
  PARTITION=Short
  CLUSTER_DIRECTIVE=""  # OUCE is a single cluster; --clusters is not valid
else
  echo "Unknown cluster. Please set the partition variable manually."
  exit 1
fi
PARTITION=${PARTITION}
WALLTIME=0:30:00
SIM_CPUS=28
CPU_MEM=8G
SIM_NAME=gwi
LOG_DIR=slurm_logs
mkdir -p ${LOG_DIR}

# Job/file naming ############################################################
# Every generated name (job name, log file, .slurm file) is built from a single
# RUN_TAG describing the configuration, using the same KEY--value convention
# that gwi.py writes into the results paths. This means a job in squeue can be
# matched to the results it produces by eye.

# Member range as a filename-safe tag. Brace expansion does NOT occur at
# assignment, so SPECIFY_ENSEMBLE_MEMBERS holds the literal string "{1..50}"
# and this operates on that text:
#   {1..50} -> 1-50 ;  {51..100} -> 51-100 ;  all -> all
MEMBER_TAG=$(echo "${SPECIFY_ENSEMBLE_MEMBERS}" | tr -d '{}' | sed 's/\.\./-/')

# Regression variables, sorted and dash-joined to match the results naming.
# gwi.py sorts its --regress-variables, so VARS=GHG,OHF,Nat becomes the
# directory VARIABLES--GHG-Nat-OHF; sorting here keeps the two in step. It also
# removes the commas, which otherwise need quoting in every command that
# touches a log file.
VARS_TAG=$(echo "${VARS}" | tr ',' '\n' | sort | paste -sd-)

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

# Compose the identifier for this job. Core fields are always present; the
# remaining flags are appended only when set away from their usual value, so
# that ordinary runs keep shorter names.
RUN_TAG="SCENARIO--${SCENARIO}"
RUN_TAG="${RUN_TAG}_VARIABLES--${VARS_TAG}"
RUN_TAG="${RUN_TAG}_REGRESSED-YEARS--${START_REGRESS}-${i}"
RUN_TAG="${RUN_TAG}_TRUNCATED-YEARS--${TRUNCATION}"
RUN_TAG="${RUN_TAG}_SAMPLES--${j}"
RUN_TAG="${RUN_TAG}_MEMBERS--${MEMBER_TAG}"
RUN_TAG="${RUN_TAG}_ITER--${count}"
if [ "${COMMITTED}" != "n" ]; then
  RUN_TAG="${RUN_TAG}_COMMITTED--${COMMITTED}"
fi
if [ "${INCLUDE_SUB_VARS}" = "y" ]; then
  RUN_TAG="${RUN_TAG}_SUB-VARS--y"
fi
if [ "${INCLUDE_RATE}" = "y" ]; then
  RUN_TAG="${RUN_TAG}_RATE--y"
fi
if [ "${CALCULATE_PRIORS_OUTPUT}" = "n" ]; then
  RUN_TAG="${RUN_TAG}_PRIORS--n"
fi

SLURM_FILE=${SIM_NAME}_${RUN_TAG}.slurm

cat > ${SLURM_FILE} << EOF
#!/bin/bash
#
## Set the maximum amount of runtime
#SBATCH --time=${WALLTIME}

## Request one node with many cpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${SIM_CPUS}
#SBATCH --mem-per-cpu=${CPU_MEM}
#SBATCH --partition=${PARTITION}
${CLUSTER_DIRECTIVE}

## Name the job and queue it
#SBATCH --job-name=${SIM_NAME}_${RUN_TAG}

## Declare an output log. %j is expanded by Slurm to the job ID, which
## keeps repeat runs of the same configuration in separate files.
#SBATCH --output=./${LOG_DIR}/${SIM_NAME}_${RUN_TAG}_JOB--%j.out

# For the single ensemble member selection runs
if [[ "${SPECIFY_ENSEMBLE_MEMBERS}" == "all" ]]; then
  # Regress against all reference temperatures at the same time
  python gwi.py --samples=${j} --regress-range=${START_REGRESS}-${i} --truncate=${TRUNCATION} --include-rate=${INCLUDE_RATE} --headline-toggles=${HEADLINE_TOGGLES} --headline-years=${HEADLINE_YEARS}  --regress-variables=${VARS} --scenario=${SCENARIO} --committed=${COMMITTED} --preindustrial-era=${PREINDUSTRIAL_ERA} --include-reg-const=${INCLUDE_REG_CONST} --specify-ensemble-member-sources-for=${SPECIFY_ENSEMBLE_MEMBER_SOURCE_FOR} --specify-ensemble-member=${SPECIFY_ENSEMBLE_MEMBERS} --include-sub-vars=${INCLUDE_SUB_VARS} --calculate-priors-output=${CALCULATE_PRIORS_OUTPUT}
else
  for k in ${SPECIFY_ENSEMBLE_MEMBERS}; do
    # Regress against each reference temperature separately
    python gwi.py --samples=${j} --regress-range=${START_REGRESS}-${i} --truncate=${TRUNCATION} --include-rate=${INCLUDE_RATE} --headline-toggles=${HEADLINE_TOGGLES} --headline-years=${HEADLINE_YEARS}  --regress-variables=${VARS} --scenario=${SCENARIO} --committed=${COMMITTED} --preindustrial-era=${PREINDUSTRIAL_ERA} --include-reg-const=${INCLUDE_REG_CONST} --specify-ensemble-member-sources-for=${SPECIFY_ENSEMBLE_MEMBER_SOURCE_FOR} --specify-ensemble-member=\$k --include-sub-vars=${INCLUDE_SUB_VARS} --calculate-priors-output=${CALCULATE_PRIORS_OUTPUT}
  done
fi


EOF

# Submit a single job to slurm.
sbatch ${SLURM_FILE}

# Remove the job file as slurm reads the script at submission time and it is
# no longer needed.
rm -rf ${SLURM_FILE}

done

# Increment the counter that keeps track of multiple runs at the same sample
# size. i.e. for each member of SUBSAMPLE_ITERATIONS, this counter will increment.
count=$((count + 1))

done
