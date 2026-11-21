#!/bin/bash
#
# Run the Global Warming Index for every configuration in a recipe: either by
# submitting Slurm jobs, or directly in this shell.
#
# Usage: activate the gwi environment, then run
#   bash run-gwi.sh <recipe> [dry|direct|slurm]
# Run it with no arguments for the full usage message.
#
# The settings live in the recipe (see recipes/TEMPLATE.sh), not here, so
# this script shouldn't need editing to run jobs. Every value taken from the
# recipe is named RECIPE_<NAME>.
#
# This script only produces the GWI results. Combine and plot them afterwards
# with combine_results_iterations.py.

# In direct mode each gwi.py run is piped through tee into its log. Without
# pipefail a failed run would look like a success, because the pipeline
# would report tee's exit status rather than gwi.py's.
set -o pipefail

# The folder holding this script, so that its helper files and recipes/ are
# found from it.
SCRIPT_DIR=$(dirname "$0")

###############################################################################
## Arguments ##################################################################

usage() {
  local recipe_list='' f
  for f in "${SCRIPT_DIR}"/recipes/*.sh; do
    [ -e "${f}" ] && recipe_list+="  ${f#./}"$'\n'
  done
  cat << EOF
Usage: bash run-gwi.sh <recipe> [mode]

Runs the Global Warming Index for every configuration in a recipe: a file
that sets every option for the runs (datasets, regression years, sample
sizes, ...). You shouldn't need to edit run-gwi.sh itself.

  <recipe>   the recipe file to run, e.g. recipes/TEMPLATE.sh
  [mode]     how to run it:
               dry     print the gwi.py commands that would run, and run
                       nothing (the default)
               direct  run each calculation now, one after another, in this
                       shell: on a laptop, or on an interactive Slurm node
               slurm   submit one Slurm job per configuration (ARC or OUCE)

Recipes available:
${recipe_list:-  (none)
}
To make a new recipe, copy the template, which lists and explains every
setting, and edit the values:
  cp recipes/TEMPLATE.sh recipes/my-recipe.sh
recipes/dev*.sh are ignored by git: use them for your own development runs.

The template is itself a quick test (well under a minute on a laptop; its
ensemble is far too small for real results):
  bash run-gwi.sh recipes/TEMPLATE.sh direct

Activate the gwi environment first (direct and slurm modes). Once the runs
have finished, combine and plot them, using the recipe's output tag:
  python combine_results_iterations.py --output-tag=<RECIPE_OUTPUT_TAG>
EOF
}

case "$1" in
  -h|--help) usage; exit 0 ;;
  '') usage >&2; exit 1 ;;
esac
if [ $# -gt 2 ]; then
  echo "Too many arguments: give a recipe, and optionally a mode." \
       "Run bash run-gwi.sh with no arguments for help." >&2
  exit 1
fi

# The recipe file to run: the first argument.
RECIPE=$1

# How to run each GWI calculation: slurm, direct or dry (see usage above).
# ${2:-dry} is the second argument, or dry when there isn't one, so that
# running a recipe never submits or runs anything by accident.
MODE=${2:-dry}

case "${MODE}" in
  slurm|direct|dry) ;;
  *) echo "Unknown mode '${MODE}': use dry, direct or slurm." >&2; exit 1 ;;
esac

if [ ! -f "${RECIPE}" ]; then
  echo "Recipe not found: ${RECIPE}." \
       "Run bash run-gwi.sh with no arguments to list the recipes." >&2
  exit 1
fi

###############################################################################
## Recipe #####################################################################

# Check the recipe against the template, which lists every setting: the
# recipe must set each of them once, and nothing else. The template is only
# the reference here; it is not read for any values. The check is in
# src/infrastructure/check_recipe.sh.
TEMPLATE=${SCRIPT_DIR}/recipes/TEMPLATE.sh
source "${SCRIPT_DIR}/src/infrastructure/check_recipe.sh"
check_recipe "${RECIPE}" "${TEMPLATE}" || exit 1

# Read the recipe: from here on, every RECIPE_<NAME> holds its value.
source "${RECIPE}"

###############################################################################
## Run ########################################################################

SIM_NAME=gwi

# Both slurm and direct mode run gwi.py with whatever python this shell has
# (Slurm jobs inherit the environment they were submitted from), so the gwi
# environment must be active. It is not activated here because each machine
# does it differently (conda on a laptop, micromamba on OUCE, a module plus
# mamba on ARC/HTC). CONDA_PREFIX is set by all three.
if [ "${MODE}" != "dry" ] && [ "$(basename "${CONDA_PREFIX:-}")" != "gwi" ]; then
  echo "The gwi environment is not active. Active environment:" \
       "${CONDA_DEFAULT_ENV:-none} (${CONDA_PREFIX:-no environment prefix set})." \
       "Activate it, e.g. with conda activate gwi or micromamba activate gwi" \
       "(on ARC, load the conda or mamba module first), and run again." \
       "To create it, see environment.yml." >&2
  exit 1
fi

if [ "${MODE}" = "slurm" ]; then
  if hostname | grep -Eq "htc|arc"; then  # ARC cluster
    PARTITION=short
    CLUSTER_DIRECTIVE="#SBATCH --clusters=${RECIPE_CLUSTER}"
  elif hostname | grep -q "ouce"; then  # OUCE cluster
    PARTITION=Short
    CLUSTER_DIRECTIVE=""  # OUCE is a single cluster; --clusters is not valid
  else
    echo "Unknown cluster. Please set the partition variable manually," \
         "or use direct mode to run in this shell instead." >&2
    exit 1
  fi
  # ${RECIPE_OUTPUT_TAG:+_${RECIPE_OUTPUT_TAG}} appends '_<tag>' only when the
  # tag is set, so an empty tag leaves the plain logs_slurm/ directory.
  LOG_DIR=logs_slurm${RECIPE_OUTPUT_TAG:+_${RECIPE_OUTPUT_TAG}}
  mkdir -p "${LOG_DIR}"
elif [ "${MODE}" = "direct" ]; then
  LOG_DIR=logs_direct${RECIPE_OUTPUT_TAG:+_${RECIPE_OUTPUT_TAG}}
  mkdir -p "${LOG_DIR}"
fi

# Keep a copy of the recipe with the logs, headed with when, where and on
# which code it was run, so the logs record what produced them.
if [ "${MODE}" != "dry" ]; then
  if GIT_COMMIT=$(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null)
  then
    if ! git -C "${SCRIPT_DIR}" diff --quiet HEAD 2>/dev/null; then
      GIT_COMMIT="${GIT_COMMIT} (with uncommitted changes to tracked files)"
    fi
  else
    GIT_COMMIT="unknown (not a git repository)"
  fi
  LOGGED_RECIPE=${LOG_DIR}/recipe_$(basename "${RECIPE}" .sh)
  LOGGED_RECIPE=${LOGGED_RECIPE}_$(date +%Y-%m-%d-%H-%M-%S).sh
  {
    echo "# Copy of ${RECIPE}, as run by run-gwi.sh in ${MODE} mode"
    echo "# Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# Host: $(hostname)"
    echo "# Git commit: ${GIT_COMMIT}"
    echo
    cat "${RECIPE}"
  } > "${LOGGED_RECIPE}"
  echo "Recipe ${RECIPE} copied to ${LOGGED_RECIPE}"
else
  echo "Dry run of ${RECIPE}: the gwi.py commands that would run." \
       "Nothing is run; add direct or slurm to run them."
fi

# Job/file naming ############################################################
# Every generated name (job name, log file, .slurm file) is built from a single
# RUN_TAG describing the configuration, using the same KEY--value convention
# that gwi.py writes into the results paths. This means a job in squeue can be
# matched to the results it produces by eye.

# Ensemble members ###########################################################
# Turn RECIPE_ERF_MEMBERS and RECIPE_GMT_MEMBERS into the runs to make: run n
# uses ERF member ERF_LIST[n] with GMT member GMT_LIST[n]. Also sets
# MEMBER_TAG for the job and log names. Stops if the combination isn't
# allowed. The logic is in src/infrastructure/prepare_ensemble_selections.sh.
source "${SCRIPT_DIR}/src/infrastructure/prepare_ensemble_selections.sh"
build_member_runs "${RECIPE_ERF_MEMBERS}" "${RECIPE_GMT_MEMBERS}" || exit 1

# Regression variables, sorted and dash-joined to match the results naming.
# gwi.py sorts its --regress-variables, so RECIPE_VARS=GHG,OHF,Nat becomes the
# directory VARIABLES--GHG-Nat-OHF; sorting here keeps the two in step. It also
# removes the commas, which otherwise need quoting in every command that
# touches a log file.
VARS_TAG=$(echo "${RECIPE_VARS}" | tr ',' '\n' | sort | paste -sd- -)

# The gwi.py call for one calculation, built here once for every mode.
# Arguments: sample size, regression end year, ERF member, GMT member.
# Sets the array GWI_COMMAND.
gwi_command() {
  GWI_COMMAND=(
    python gwi.py
    --samples="$1"
    --seed="${RECIPE_SEED}"
    --regress-range="${RECIPE_START_REGRESS}-$2"
    --truncate="${RECIPE_TRUNCATION}"
    --include-rate="${RECIPE_INCLUDE_RATE}"
    --headline-toggles="${RECIPE_HEADLINE_TOGGLES}"
    --headline-years="${RECIPE_HEADLINE_YEARS}"
    --regress-variables="${RECIPE_VARS}"
    --erf-dataset="${RECIPE_ERF_DATASET}"
    --gmt-dataset="${RECIPE_GMT_DATASET}"
    --picontrol-dataset="${RECIPE_PICONTROL_DATASET}"
    --committed="${RECIPE_COMMITTED}"
    --preindustrial-era="${RECIPE_PREINDUSTRIAL_ERA}"
    --include-reg-const="${RECIPE_INCLUDE_REG_CONST}"
    --erf-member="$3"
    --gmt-member="$4"
    --include-sub-vars="${RECIPE_INCLUDE_SUB_VARS}"
    --calculate-priors-output="${RECIPE_CALCULATE_PRIORS_OUTPUT}"
    --output-tag="${RECIPE_OUTPUT_TAG}"
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
for j in "${RECIPE_SUBSAMPLE_ITERATIONS[@]}"
do

for i in ${RECIPE_END_REGRESS}
do

echo "Iteration ${count}: ${j} samples, regressed years ${RECIPE_START_REGRESS}-${i}"

# Compose the identifier for this job. Core fields are always present; the
# remaining flags are appended only when set away from their usual value, so
# that ordinary runs keep shorter names. This is needed, not just tidier: the
# identifier goes into the log file name, and file names are limited to 255
# bytes (macOS and Linux), which a name with every field in it can exceed.
# Every setting is in the log itself anyway, in the gwi.py command at its top.
RUN_TAG="ERF--${RECIPE_ERF_DATASET}_GMT--${RECIPE_GMT_DATASET}"
RUN_TAG="${RUN_TAG}_VARIABLES--${VARS_TAG}"
RUN_TAG="${RUN_TAG}_REGRESSED-YEARS--${RECIPE_START_REGRESS}-${i}"
RUN_TAG="${RUN_TAG}_TRUNCATED-YEARS--${RECIPE_TRUNCATION}"
RUN_TAG="${RUN_TAG}_SAMPLES--${j}"
RUN_TAG="${RUN_TAG}_MEMBERS--${MEMBER_TAG}"
RUN_TAG="${RUN_TAG}_ITER--${count}"
if [ "${RECIPE_PREINDUSTRIAL_ERA}" != "1850-1900" ]; then
  RUN_TAG="${RUN_TAG}_PREINDUSTRIAL--${RECIPE_PREINDUSTRIAL_ERA}"
fi
if [ "${RECIPE_COMMITTED}" != "n" ]; then
  RUN_TAG="${RUN_TAG}_COMMITTED--${RECIPE_COMMITTED}"
fi
if [ "${RECIPE_INCLUDE_SUB_VARS}" = "y" ]; then
  RUN_TAG="${RUN_TAG}_SUB-VARS--y"
fi
if [ "${RECIPE_INCLUDE_RATE}" = "y" ]; then
  RUN_TAG="${RUN_TAG}_RATE--y"
fi
if [ "${RECIPE_CALCULATE_PRIORS_OUTPUT}" = "n" ]; then
  RUN_TAG="${RUN_TAG}_PRIORS--n"
fi
if [ -n "${RECIPE_SEED}" ]; then
  RUN_TAG="${RUN_TAG}_SEED--${RECIPE_SEED}"
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
  cat > "${SLURM_FILE}" << EOF
#!/bin/bash
#
## Set the maximum amount of runtime
#SBATCH --time=${RECIPE_WALLTIME}

## Request one node with many cpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${RECIPE_SIM_CPUS}
#SBATCH --mem-per-cpu=${RECIPE_CPU_MEM}
#SBATCH --partition=${PARTITION}
${CLUSTER_DIRECTIVE}

## Name the job and queue it
#SBATCH --job-name=${SIM_NAME}_${RUN_TAG}

## Declare an output log. %j is expanded by Slurm to the job ID, which
## keeps repeat runs of the same configuration in separate files.
#SBATCH --output=./${LOG_DIR}/${SIM_NAME}_${RUN_TAG}_JOB--%j.out

## One gwi.py run for each selected member, one after another; a single run
## when RECIPE_ERF_MEMBERS and RECIPE_GMT_MEMBERS are both all.
EOF
  # Second step: append one gwi.py command line per member to the end of the
  # job file (>> appends), after the header written above. The job runs
  # these lines in order.
  # One gwi.py run per (ERF member, GMT member) pair: the lists line up by n
  for n in "${!ERF_LIST[@]}"; do
    gwi_command "${j}" "${i}" "${ERF_LIST[n]}" "${GMT_LIST[n]}"
    print_gwi_command >> "${SLURM_FILE}"
  done

  # Submit a single job to slurm.
  sbatch "${SLURM_FILE}"

  # Remove the job file as slurm reads the script at submission time and it
  # is no longer needed.
  rm -f "${SLURM_FILE}"
fi

done

# Increment the counter that keeps track of multiple runs at the same sample
# size. i.e. for each member of RECIPE_SUBSAMPLE_ITERATIONS, this counter will
# increment.
count=$((count + 1))

done
