#!/bin/bash
#
# Recipe template for run-gwi.sh: every setting, with what it does and the
# values it can take.
#
# A recipe is a list of settings for run-gwi.sh. To make one, copy this file
# and edit the values:
#
#     cp recipes/TEMPLATE.sh recipes/my-recipe.sh
#
# then run it with
#
#     bash run-gwi.sh recipes/my-recipe.sh [dry|direct|slurm]
#
# A recipe sets every RECIPE_ setting listed here, and nothing else:
# run-gwi.sh checks this before running anything, and stops if a setting is
# missing, repeated, or not one of these. The comments are optional, so a
# recipe can be just the list of settings. recipes/dev*.sh are ignored by git:
# use them for your own development runs.
#
# As it stands, this file is itself a quick test: one regression end year and
# 10,000 members, which takes well under a minute in direct mode.
# WARNING: its results are not fit for any real use. The ensemble is
# trivially small, so it only checks that everything runs; real results need
# much larger ensembles (see RECIPE_SAMPLING_ITERATIONS).
#
# Adding a new setting: add it here, with its explanation, and to every
# recipe in recipes/.

###############################################################################
## Output #####################################################################

# A label that keeps this piece of work's output away from everything else's:
# logs go to logs_slurm_<TAG>/ (or logs_direct_<TAG>/ in direct mode),
# results to results_<TAG>/ and plots to plots_<TAG>/. Use it to keep separate
# projects apart -- Thorne et al, IGCC, a collaborator's runs, your own
# development -- so that they neither overwrite each other nor all get swept
# up every time the combination script runs. Usually the recipe's own name.
# Recipes that make up one experiment share a tag, so that the combination
# script processes them together:
#
#     python combine_results_iterations.py --output-tag=<TAG>
#
# Alternatively, give it any one of the experiment's recipes, and it reads
# the tag from there:
#
#     python combine_results_iterations.py --recipe=<recipe>
#
# run-gwi.sh prints the command when it finishes.
#
# Leave empty for the plain logs_slurm/ (or logs_direct/), results/ and plots/
# folders. The option is always passed to gwi.py and its value says whether
# it applies, the same convention as COMMITTED=n or INCLUDE_SUB_VARS=n.
# e.g. RECIPE_OUTPUT_TAG=Thorne2025
# e.g. RECIPE_OUTPUT_TAG=IGCC2025
# e.g. RECIPE_OUTPUT_TAG=          (no tag)
RECIPE_OUTPUT_TAG=quick-test

###############################################################################
## GWI method settings ########################################################

# Select the date to start the regression range from
RECIPE_START_REGRESS=1850

# Select the date to end the regression range at. Create a sequential range
# for the range of regressed years.
# e.g. RECIPE_END_REGRESS=2025                  (a single end year: the
#                                                full-information GWI)
# e.g. RECIPE_END_REGRESS=$(seq 2000 2025)      (every end year from 2000 to
#                                                2025, inclusive: for the
#                                                historical-only GWI)
RECIPE_END_REGRESS=2025

# Select how to sample the ensemble: one iteration for each entry, each a
# sampling design and the ensemble size N (the members to rebuild) joined by
# a dash, or all. The complete ensemble is every combination of the members
# of the four sources of uncertainty (FaIR parameter sets, ERF members,
# observed members, piControl segments): about 790 million members for
# IGCC-2025 with HadCRUT-2025. The designs, each treating the four sources
# alike:
#   latin   N members using every member of each source equally often, the
#           sources paired at random (a Latin hypercube), and no member
#           drawn twice until every member has been: at N = the complete
#           ensemble's size it is the complete ensemble
#   mc      N members drawn independently from the complete ensemble,
#           with replacement (Monte Carlo)
#   cube    every combination of k random members of each source, with the
#           largest k that keeps it within N (so it can fall short of N);
#           earlier GWI releases used a cube that kept every parameter set
#   all     every member of the complete ensemble: no sampling error, the
#           same every time; needs a large-memory node for IGCC-sized
#           datasets (see the README's Memory and Computing Resources)
# N is a whole number, or in e notation: 1e6 is 1,000,000.
# e.g. RECIPE_SAMPLING_ITERATIONS=(latin-1e4)           (a quick test)
# e.g. RECIPE_SAMPLING_ITERATIONS=(latin-1e6)           (a routine run)
# e.g. RECIPE_SAMPLING_ITERATIONS=(latin-1e6 latin-1e6) (two repeat
#                                                        iterations, averaged
#                                                        by the combination
#                                                        script)
# e.g. RECIPE_SAMPLING_ITERATIONS=(latin-1e5 latin-1e6 latin-1e7)
#                                                       (scaling up)
# e.g. RECIPE_SAMPLING_ITERATIONS=(all)                 (every member)
RECIPE_SAMPLING_ITERATIONS=(latin-1e4)

# Select a seed for the random sampling of ensemble members, for exactly
# reproducible runs: runs of the same configuration on the same inputs then
# give identical results (e.g. to check that a code change leaves the results
# unchanged). Leave empty for the usual setting, in which every run draws
# its own seed, and so samples afresh; the seed drawn is logged, and named
# in the results files, so that any run can be reproduced.
# With a seed set, each iteration in RECIPE_SAMPLING_ITERATIONS uses the
# seed plus its number less one (42, 43, 44, ... for RECIPE_SEED=42), so
# repeat iterations draw different members; the end years of one iteration
# share its seed, and so its members.
# e.g. RECIPE_SEED=     (unseeded; the usual setting)
# e.g. RECIPE_SEED=42
RECIPE_SEED=

# Select the reference period for the temperature datasets
# The selected period offset applies to FaIR outputs, GMT Observations,
# and piControl internal variability.
# Both years are included, as in IPCC and WMO usage: 1850-1900 is the 51
# years from 1850 to 1900, and 1981-2010 the 30 years from 1981 to 2010.
# 1850-1900 is the IPCC's approximation of pre-industrial levels: "the
# 51-year period 1850-1900" (IPCC SR1.5, Chapter 1, Section 1.2.1).
# Note that 'n' can be used to disable this preprocessing offset: the FaIR
# outputs and GMT observations then keep their own levels (e.g. absolute
# temperatures, around 14 degrees C), so include the regression constant
# (RECIPE_INCLUDE_REG_CONST=y) to take up the difference. The piControl
# segments are always anomalies (internal variability is a fluctuation about
# zero): without this offset, each is offset by its own mean.
# e.g. 1850-1900   (pre-industrial, as defined by the IPCC)
# e.g. 1981-2010
RECIPE_PREINDUSTRIAL_ERA=1850-1900

# Select whether to include a constant term offset in the multi-variable regression
# e.g. y (include the constant)
# e.g. n (do not include the constant term)
RECIPE_INCLUDE_REG_CONST=n

# Select which variables to regress on.
# e.g. GHG,OHF,Nat
# e.g. Ant,Nat
# e.g. Tot
RECIPE_VARS=GHG,OHF,Nat

# Select whether to include sub-variables in the output
# i.e. this will calculate component-wise contributions to the aggregate
# variables (e.g. GHG = CO2 + CH4 + N2O + F-gases).
# e.g. y
# e.g. n
RECIPE_INCLUDE_SUB_VARS=n

# Select the input datasets. The ERFs and the reference temperatures (GMT) are
# chosen separately: one of each per gwi.py run.
# - Observed datasets (filed in data/observed-<YYYY>/) carry the year their
#   history ends. Any observed ERF can be paired with any observed GMT, as
#   long as both cover the regressed years.
# - Simulated datasets (climate model large ensembles, filed in
#   data/<experiment>/<ERF or GMT>/<model>/) pair only with themselves: set
#   RECIPE_ERF_DATASET and RECIPE_GMT_DATASET to the same simulated dataset.
# In the lists below, the parts in <angle brackets> are chosen from the values
# listed under them; the rest of each name is fixed.

# Select the ERF dataset, from:
#   IGCC-<YYYY>
#     <YYYY> = 2022, 2023, 2024, 2025
#     IGCC historical ERFs, 1000 members
#     (RECIPE_INCLUDE_SUB_VARS=y needs 2024 or 2025)
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
RECIPE_ERF_DATASET=IGCC-2025

# Select which ERF ensemble members to use
# e.g. all                     (sample from the whole ensemble; the usual)
# e.g. 7                       (a single member)
# e.g. {1..60}                 (each member in turn, one gwi.py run each)
# e.g. paired_with_GMT_member  (each member listed in RECIPE_GMT_MEMBERS in
#                               turn, with the ERF member of the same label)
# NOTE: IGCC and SSP_extension members are labelled by configuration number
# (ens<number> for IGCC), so use all for them; likewise for the datasets with
# a single ERF timeseries.
RECIPE_ERF_MEMBERS=all

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
RECIPE_GMT_DATASET=HadCRUT-2025

# Select which GMT ensemble members to use
# e.g. all                     (sample from the whole ensemble; the usual)
# e.g. 7                       (a single member)
# e.g. {1..200}                (each member in turn, one gwi.py run each)
# e.g. paired_with_ERF_member  (each member listed in RECIPE_ERF_MEMBERS in
#                               turn, with the GMT member of the same label)
RECIPE_GMT_MEMBERS=all

# How RECIPE_ERF_MEMBERS and RECIPE_GMT_MEMBERS combine (the CASEs in
# src/infrastructure/prepare_ensemble_selections.sh):
# - CASE 1: all and all: one run, sampling from both whole ensembles.
# - CASE 2: members listed on one side, all on the other (e.g.
#   RECIPE_ERF_MEMBERS=all, RECIPE_GMT_MEMBERS={1..200}): one run per listed
#   member, each sampling from the other dataset's whole ensemble.
# - CASE 3: members listed on one side, the pairing keyword on the other
#   (e.g. RECIPE_ERF_MEMBERS=paired_with_GMT_member,
#   RECIPE_GMT_MEMBERS={1..60}): one run per listed member, with ERF and GMT
#   members of the same label. Required for FutureVolc_ESM-rcp45-Volc.
# Anything else stops with an error before any run starts:
# - members listed on both sides, without a pairing keyword (it is unclear
#   whether to pair them or to run every combination);
# - pairing keywords on both sides (there would be no members to pair);
# - a pairing keyword with all on the other side (all can't be paired);
# - a pairing keyword on the wrong side (e.g.
#   RECIPE_ERF_MEMBERS=paired_with_ERF_member).

# Select the piControl dataset that samples internal variability
# e.g. CMIP6  (the only one for now)
RECIPE_PICONTROL_DATASET=CMIP6

# Select whether to consider committed warming at constant ERF.
# This amends/extends the scenario to hold ERF constant from a start year
# to an end year.
# Format: start_year-end_year
# You can use 'end_regress' as a keyword for the start year.
# e.g. n (no committed warming)
# e.g. 2025-2300 (constant ERF from 2025 to 2300)
# e.g. end_regress-2300 (constant ERF from the end of regression to 2300)
RECIPE_COMMITTED=n

# Select truncation range
RECIPE_TRUNCATION=1850-2050

# The warming definitions to give: at least one. Each gets its own
# timeseries CSV (GWI_results_timeseries-<DEFINITION>), covering every year
# for which it can be produced from the underlying annual timeseries (the
# years whose whole window lies within the truncation range), and its values
# at the headline years go into the headlines CSV. The annual timeseries is
# always written, since the plots build on it.
#   annual  the warming in each year
#   SR1.5   the value at mid-year of the line fitted through the last 15 years
#   AR6     the mean of the last 10 years
#   CGWL    the mean of the 19 years centred on the year (needs forcing after it)
#   rates   the slope of the line fitted through the last 10 years (AR6 rate)
# e.g. 'annual,SR1.5,AR6,CGWL,rates'
# e.g. annual, for the basic annual timeseries
RECIPE_DEFINITIONS='annual,SR1.5,AR6,CGWL,rates'

# Select which years for the headlines to cover.
# e.g. 'end_regress' for the end of the regressed range
# e.g. 'end_trunc' for the end of the truncation range
# e.g. 'IGCC' for the end of the regressed range, plus each definition's
#      year assessed in the IPCC reports (2017 for annual and SR1.5,
#      2010-2019 for AR6 and rates), repeated to validate against them
# e.g. '2024' for a single year
# e.g. '2023,2024,2025' for multiple separate years.
# e.g. 'end_regress,2050,2100,2300' to combine end_regress and manual years
# e.g. $(seq -s, 1950 2025) will create a comma-separated list of years
RECIPE_HEADLINE_YEARS='end_regress'

# The quantiles: every definition's timeseries as percentiles every this
# many percent, from the 0th to the 100th, written to one netCDF file
# (GWI_results_timeseries-quantiles), beside the summary CSVs (the median
# and the likely and very likely ranges, which are always written). They
# show the whole distribution finely: the sampling studies compare them,
# and exceedance probabilities in each year can be read off them. Taken in
# the same pass as the summary, at 0.1 they make the whole run about 1.25
# times as long at a million members, and 1.6 times at ten million (latin,
# every definition, 28 CPUs: 59 s to 73 s, 3.6 to 5.7 minutes); the file
# grows with the number of levels, not the number of members (13 MB for
# every definition at 0.1).
# e.g. 0.1   every 0.1 percentile (1001 levels)
# e.g. 1     every percentile (101 levels)
# e.g. none  no quantiles: the summary CSVs only (also if left empty)
RECIPE_QUANTILE_STEP=0.1

###############################################################################
## Slurm settings (slurm mode only) ###########################################

# Select which Slurm cluster to submit to.
# Oxford ARC runs two clusters under one Slurm accounting database: 'arc'
# (large nodes, 48+ cores) and 'htc' (smaller nodes, GPUs). List them with
# `sacctmgr show cluster` or `sinfo -M all`.
# Jobs are assigned to a cluster once, at submission time, and never migrate
# afterwards; without --clusters, sbatch submits to whichever cluster you
# happen to be logged into. Not used on OUCE, which is a single cluster.
# e.g. arc      (force ARC)
# e.g. htc      (force HTC)
# e.g. arc,htc  (let sbatch pick whichever offers the earliest start time)
RECIPE_CLUSTER=htc

# Select the resources for each Slurm job: time limit, CPUs, and memory per
# CPU. 0:10:00, 28 and 9000M suit about 6 million members with every
# definition (3 to 5 minutes, about 1.5 times that with the quantiles at
# 0.1, peaking near 5 GiB); the time and memory grow with the number of
# members, and member-by-member runs need more time (see the README's Memory
# and Computing Resources). Roughly, on 28 CPUs with every definition and
# the quantiles at 0.1: latin-1e6 1.5 minutes, latin-1e7 6 minutes,
# latin-1e8 50 minutes; all (790 million members for IGCC-2025) about 6
# hours, and only on a large-memory node: about 160 GB, for blocks of 4
# years (with less, the blocks shrink to 1 year and the run is several
# times slower).
RECIPE_WALLTIME=0:10:00
RECIPE_SIM_CPUS=28
RECIPE_CPU_MEM=9000M
