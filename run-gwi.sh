#!/bin/bash
#
# Run the Global Warming Index for every configuration set below: either by
# submitting Slurm jobs, or directly in this shell.
#
# Usage: edit the settings below, activate the gwi environment, then run
#   bash run-gwi.sh
#
# This script only produces the GWI results. Combine and plot them afterwards
# with combine_results_iterations.py.

# In direct mode each gwi.py run is piped through tee into its log. Without
# pipefail a failed run would look like a success, because the pipeline
# would report tee's exit status rather than gwi.py's.
set -o pipefail

###############################################################################
## How to run #################################################################

# Select how to run each GWI calculation.
# e.g. slurm   (submit one Slurm job per configuration to the queue)
# e.g. direct  (run each calculation now, one after another, in this shell:
#               on a laptop, or on an interactive Slurm node)
# e.g. dry     (print the gwi.py commands that would run, and run nothing)
MODE=slurm

# Optional label that keeps this piece of work's output away from everything
# else's: logs go to logs_slurm_<TAG>/ (or logs_direct_<TAG>/ in direct mode),
# results to results_<TAG>/ and plots to plots_<TAG>/. Use it to keep separate
# projects apart -- Thorne et al, IGCC, a collaborator's runs, your own
# development -- so that they neither overwrite each other nor all get swept
# up every time the combination script runs. Pass the same tag to
# combine_results_iterations.py:
#
#     python combine_results_iterations.py --output-tag=${OUTPUT_TAG}
#
# Leave empty for the plain logs_slurm/ (or logs_direct/), results/ and plots/
# folders. The option is always passed to gwi.py and its value says whether
# it applies, the same convention as COMMITTED=n or INCLUDE_SUB_VARS=n.
# e.g. OUTPUT_TAG=Thorne2025
# e.g. OUTPUT_TAG=IGCC2025
# e.g. OUTPUT_TAG=          (no tag)
OUTPUT_TAG=output-tag-test

###############################################################################
## GWI method settings ########################################################

# Select the date to start the regression range from
START_REGRESS=1850

# Select the date to end the regression range at. Create a sequential range
# for the range of regressed years.
# This is for calculating the historical-only GWI:
# END_REGRESS=`seq 2000 2023`  # This is inclusive of the start and end years
# This is for calculating the GWI with all years:
END_REGRESS=`seq 2020 2025`

# Create array of subsampling sizes to calculate.
# This is for scaling up the calculation:
# SUBSAMPLE_ITERATIONS=(60 65 70 75 80 85 90 95 100)  # Size of subsampling
# This is for repeating final calculations at one size:
# SUBSAMPLE_ITERATIONS=(10000)  # Size of subsampling to max out the calculation
SUBSAMPLE_ITERATIONS=(60 60 60)  # Size of subsampling

# Select a seed for the random sampling of ensemble members, for exactly
# reproducible runs: runs of the same configuration on the same inputs then
# give identical results (e.g. to check that a code change leaves the results
# unchanged). Leave empty for the usual, unseeded sampling, in which every run
# samples afresh.
# NOTE: every run uses the same seed, so with a seed set, repeat iterations
# of the same configuration in SUBSAMPLE_ITERATIONS give identical results.
# e.g. SEED=     (unseeded; the usual setting)
# e.g. SEED=42
SEED=

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

# Select the input datasets. The ERFs and the reference temperatures (GMT) are
# chosen separately: one of each per gwi.py run.
# - Observed datasets (filed in data/observed-<YYYY>/) carry the year their
#   history ends. Any observed ERF can be paired with any observed GMT, as
#   long as both cover the regressed years.
# - Simulated datasets (climate model large ensembles, filed in
#   data/<experiment>/<ERF or GMT>/<model>/) pair only with themselves: set
#   ERF_DATASET and GMT_DATASET to the same simulated dataset.
# In the lists below, the parts in <angle brackets> are chosen from the values
# listed under them; the rest of each name is fixed.

# Select the ERF dataset, from:
#   IGCC-<YYYY>
#     <YYYY> = 2022, 2023, 2024, 2025
#     IGCC historical ERFs, 1000 members
#     (INCLUDE_SUB_VARS=y needs 2024 or 2025)
#   SSP_extension-2022-<SSP>
#     <SSP> = SSP119, SSP126, SSP245, SSP370, SSP434, SSP460, SSP534-over, SSP585
#     observed history to 2022, then <SSP> to 2100; 841 members. Its historical
#     ERFs are not the IGCC ERFs: they are generated differently, and differ
#     before 2022 too.
#   SMILE_ESM-<SSP>
#     <SSP> = SSP126, SSP245, SSP370
#     MPI-ESM1.2-LR large ensemble: historical to 2014, then <SSP>; a single
#     ERF timeseries
#   FutureVolc_ESM-rcp45-<Volcanism>
#     <Volcanism> = Volc, VolcConst
#     NorESM1-M large ensemble: historical to 2005, then RCP4.5 with
#     stochastic future volcanic eruptions (Volc: one natural ERF member for
#     each GMT member, 1 to 60) or constant background volcanic forcing
#     (VolcConst: a single ERF timeseries)
# e.g. IGCC-2025
ERF_DATASET=IGCC-2025

# Select which ERF ensemble members to use
# e.g. all                     (sample from the whole ensemble; the usual)
# e.g. 7                       (a single member)
# e.g. {1..60}                 (each member in turn, one gwi.py run each)
# e.g. paired_with_GMT_member  (each member listed in GMT_MEMBERS in turn,
#                               with the ERF member of the same label)
# NOTE: IGCC and SSP_extension members are labelled by configuration number
# (ens<number> for IGCC), so use all for them; likewise for the datasets with
# a single ERF timeseries.
ERF_MEMBERS=all

# Select the GMT (reference temperature) dataset, from:
#   HadCRUT-<YYYY>
#     <YYYY> = 2022, 2023, 2024, 2025
#     HadCRUT5 ensemble, members 1 to 200
#   JK_multi-<YYYY>
#     <YYYY> = 2024, 2025
#     John Kennedy's multi-dataset ensemble, members 1 to 100
#   SMILE_ESM-<SSP>
#     <SSP> = SSP126, SSP245, SSP370
#     MPI-ESM1.2-LR large ensemble: historical to 2014, then <SSP>; members
#     0 to 49
#   FutureVolc_ESM-rcp45-<Volcanism>
#     <Volcanism> = Volc, VolcConst
#     NorESM1-M large ensemble: historical to 2005, then RCP4.5 with
#     stochastic future volcanic eruptions (Volc) or constant background
#     volcanic forcing (VolcConst); members 1 to 60
# e.g. HadCRUT-2025
GMT_DATASET=HadCRUT-2025

# Select which GMT ensemble members to use
# e.g. all                     (sample from the whole ensemble; the usual)
# e.g. 7                       (a single member)
# e.g. {1..200}                (each member in turn, one gwi.py run each)
# e.g. paired_with_ERF_member  (each member listed in ERF_MEMBERS in turn,
#                               with the GMT member of the same label)
GMT_MEMBERS=all

# How ERF_MEMBERS and GMT_MEMBERS combine (the CASEs in
# src/prepare_ensemble_selections.sh):
# - CASE 1: all and all: one run, sampling from both whole ensembles.
# - CASE 2: members listed on one side, all on the other (e.g.
#   ERF_MEMBERS=all, GMT_MEMBERS={1..200}): one run per listed member, each
#   sampling from the other dataset's whole ensemble.
# - CASE 3: members listed on one side, the pairing keyword on the other
#   (e.g. ERF_MEMBERS=paired_with_GMT_member, GMT_MEMBERS={1..60}): one run
#   per listed member, with ERF and GMT members of the same label. Required
#   for FutureVolc_ESM-rcp45-Volc.
# Anything else stops with an error before any run starts:
# - members listed on both sides, without a pairing keyword (it is unclear
#   whether to pair them or to run every combination);
# - pairing keywords on both sides (there would be no members to pair);
# - a pairing keyword with all on the other side (all can't be paired);
# - a pairing keyword on the wrong side (e.g. ERF_MEMBERS=paired_with_ERF_member).

# Select the piControl dataset that samples internal variability
# e.g. CMIP6  (the only one for now)
PICONTROL_DATASET=CMIP6

# Select whether to consider committed warming at constant ERF.
# This amends/extends the scenario to hold ERF constant from a start year
# to an end year.
# Format: start_year-end_year
# You can use 'end_regress' as a keyword for the start year.
# e.g. n (no committed warming)
# e.g. 2025-2300 (constant ERF from 2025 to 2300)
# e.g. end_regress-2300 (constant ERF from the end of regression to 2300)
COMMITTED=n

# Select truncation range
TRUNCATION=1850-2050

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
# e.g. $(seq -s, 1950 2025) will create a comma-separated list of years
HEADLINE_YEARS='end_regress'

# Quick test ##################################################################
# A small, fast configuration for checking that everything runs, e.g. on a
# laptop with MODE=direct. Uncomment to override the settings above.
# END_REGRESS=2025
# SUBSAMPLE_ITERATIONS=(10)
# OUTPUT_TAG=quick-test


###############################################################################
## Slurm settings (slurm mode only) ###########################################

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

WALLTIME=0:10:00
SIM_CPUS=28
CPU_MEM=9000M


###############################################################################
## Run ########################################################################

case "${MODE}" in
  slurm|direct|dry) ;;
  *) echo "Unknown MODE '${MODE}': use slurm, direct or dry." >&2; exit 1 ;;
esac

SIM_NAME=gwi

# Both slurm and direct mode run gwi.py with whatever python this shell has
# (Slurm jobs inherit the environment they were submitted from), so the gwi
# environment must be active. It is not activated here because each machine
# does it differently (conda on a laptop, micromamba on OUCE, a module plus
# mamba on ARC/HTC). CONDA_PREFIX is set by all three.
if [ "${MODE}" != "dry" ] && [ "$(basename "${CONDA_PREFIX:-}")" != "gwi" ]; then
  echo "The gwi environment is not active. Active environment:" \
       "${CONDA_DEFAULT_ENV:-none} (${CONDA_PREFIX:-no environment prefix set})." \
       "Activate gwi, or create it from environment.yml, and run again." >&2
  exit 1
fi

if [ "${MODE}" = "slurm" ]; then
  if hostname | grep -Eq "htc|arc"; then  # ARC cluster
    PARTITION=short
    CLUSTER_DIRECTIVE="#SBATCH --clusters=${CLUSTER}"
  elif hostname | grep -q "ouce"; then  # OUCE cluster
    PARTITION=Short
    CLUSTER_DIRECTIVE=""  # OUCE is a single cluster; --clusters is not valid
  else
    echo "Unknown cluster. Please set the partition variable manually," \
         "or set MODE=direct to run in this shell instead." >&2
    exit 1
  fi
  # ${OUTPUT_TAG:+_${OUTPUT_TAG}} appends '_<tag>' only when OUTPUT_TAG is
  # set, so an empty tag leaves the plain logs_slurm/ directory.
  LOG_DIR=logs_slurm${OUTPUT_TAG:+_${OUTPUT_TAG}}
  mkdir -p "${LOG_DIR}"
elif [ "${MODE}" = "direct" ]; then
  LOG_DIR=logs_direct${OUTPUT_TAG:+_${OUTPUT_TAG}}
  mkdir -p "${LOG_DIR}"
fi

# Job/file naming ############################################################
# Every generated name (job name, log file, .slurm file) is built from a single
# RUN_TAG describing the configuration, using the same KEY--value convention
# that gwi.py writes into the results paths. This means a job in squeue can be
# matched to the results it produces by eye.

# Ensemble members ###########################################################
# Turn ERF_MEMBERS and GMT_MEMBERS into the runs to make: run n uses ERF
# member ERF_LIST[n] with GMT member GMT_LIST[n]. Also sets MEMBER_TAG for the
# job and log names. Stops if the combination isn't allowed. The logic is in
# src/prepare_ensemble_selections.sh.
source "$(dirname "$0")/src/prepare_ensemble_selections.sh"
build_member_runs "${ERF_MEMBERS}" "${GMT_MEMBERS}" || exit 1

# Regression variables, sorted and dash-joined to match the results naming.
# gwi.py sorts its --regress-variables, so VARS=GHG,OHF,Nat becomes the
# directory VARIABLES--GHG-Nat-OHF; sorting here keeps the two in step. It also
# removes the commas, which otherwise need quoting in every command that
# touches a log file.
VARS_TAG=$(echo "${VARS}" | tr ',' '\n' | sort | paste -sd- -)

# The gwi.py call for one calculation, built here once for every mode.
# Arguments: sample size, regression end year, ERF member, GMT member.
# Sets the array GWI_COMMAND.
gwi_command() {
  GWI_COMMAND=(
    python gwi.py
    --samples="$1"
    --seed="${SEED}"
    --regress-range="${START_REGRESS}-$2"
    --truncate="${TRUNCATION}"
    --include-rate="${INCLUDE_RATE}"
    --headline-toggles="${HEADLINE_TOGGLES}"
    --headline-years="${HEADLINE_YEARS}"
    --regress-variables="${VARS}"
    --erf-dataset="${ERF_DATASET}"
    --gmt-dataset="${GMT_DATASET}"
    --picontrol-dataset="${PICONTROL_DATASET}"
    --committed="${COMMITTED}"
    --preindustrial-era="${PREINDUSTRIAL_ERA}"
    --include-reg-const="${INCLUDE_REG_CONST}"
    --erf-member="$3"
    --gmt-member="$4"
    --include-sub-vars="${INCLUDE_SUB_VARS}"
    --calculate-priors-output="${CALCULATE_PRIORS_OUTPUT}"
    --output-tag="${OUTPUT_TAG}"
  )
}

# Print the command in GWI_COMMAND as one line that the shell can run back.
# Arguments made of ordinary characters are printed as they are; anything
# else (a space, say) is quoted.
print_gwi_command() {
  local arg line=''
  for arg in "${GWI_COMMAND[@]}"; do
    if [[ "${arg}" =~ ^[A-Za-z0-9_./,=:+@%-]*$ ]]; then
      line+="${arg} "
    else
      line+="$(printf '%q' "${arg}") "
    fi
  done
  echo "${line% }"
}

# Keep track of which iteration we are on (avoid overwriting log files)
count=1
# Run, or submit, each configuration
for j in "${SUBSAMPLE_ITERATIONS[@]}"
do

for i in $END_REGRESS
do

echo "Iteration ${count}: ${j} samples, regressed years ${START_REGRESS}-${i}"

# Compose the identifier for this job. Core fields are always present; the
# remaining flags are appended only when set away from their usual value, so
# that ordinary runs keep shorter names.
RUN_TAG="ERF--${ERF_DATASET}_GMT--${GMT_DATASET}"
RUN_TAG="${RUN_TAG}_VARIABLES--${VARS_TAG}"
RUN_TAG="${RUN_TAG}_REGRESSED-YEARS--${START_REGRESS}-${i}"
RUN_TAG="${RUN_TAG}_TRUNCATED-YEARS--${TRUNCATION}"
RUN_TAG="${RUN_TAG}_SAMPLES--${j}"
RUN_TAG="${RUN_TAG}_MEMBERS--${MEMBER_TAG}"
RUN_TAG="${RUN_TAG}_ITER--${count}"
if [ "${PREINDUSTRIAL_ERA}" != "1850-1900" ]; then
  RUN_TAG="${RUN_TAG}_PREINDUSTRIAL--${PREINDUSTRIAL_ERA}"
fi
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
if [ -n "${SEED}" ]; then
  RUN_TAG="${RUN_TAG}_SEED--${SEED}"
fi

if [ "${MODE}" = "dry" ]; then
  # One gwi.py run per (ERF member, GMT member) pair: the lists line up by n
  for n in "${!ERF_LIST[@]}"; do
    gwi_command "${j}" "${i}" "${ERF_LIST[n]}" "${GMT_LIST[n]}"
    print_gwi_command
  done

elif [ "${MODE}" = "direct" ]; then
  # One log per configuration, holding every member's run, as for a Slurm
  # job. The timestamp keeps repeat runs of the same configuration apart, as
  # the job ID does in slurm mode.
  LOG_FILE=${LOG_DIR}/${SIM_NAME}_${RUN_TAG}_DIRECT--$(date +%Y-%m-%d-%H-%M-%S).out
  # One gwi.py run per (ERF member, GMT member) pair: the lists line up by n
  for n in "${!ERF_LIST[@]}"; do
    gwi_command "${j}" "${i}" "${ERF_LIST[n]}" "${GMT_LIST[n]}"
    print_gwi_command | tee -a "${LOG_FILE}"
    if ! "${GWI_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
      echo "gwi.py failed; stopping. Log: ${LOG_FILE}" >&2
      exit 1
    fi
  done

else  # slurm
  # One Slurm job per configuration, holding every member's gwi.py run, one
  # after another (the same grouping as before).
  SLURM_FILE=${SIM_NAME}_${RUN_TAG}.slurm

  # Write the job file in two steps: first the #SBATCH header, here, then
  # the gwi.py command lines, appended below.
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

## One gwi.py run for each selected member, one after another; a single run
## when ERF_MEMBERS and GMT_MEMBERS are both all.
EOF
  # Second step: append one gwi.py command line per member to the end of the
  # job file (>> appends), after the header written above. The job runs
  # these lines in order.
  # One gwi.py run per (ERF member, GMT member) pair: the lists line up by n
  for n in "${!ERF_LIST[@]}"; do
    gwi_command "${j}" "${i}" "${ERF_LIST[n]}" "${GMT_LIST[n]}"
    print_gwi_command >> ${SLURM_FILE}
  done

  # Submit a single job to slurm.
  sbatch ${SLURM_FILE}

  # Remove the job file as slurm reads the script at submission time and it
  # is no longer needed.
  rm -f ${SLURM_FILE}
fi

done

# Increment the counter that keeps track of multiple runs at the same sample
# size. i.e. for each member of SUBSAMPLE_ITERATIONS, this counter will increment.
count=$((count + 1))

done
