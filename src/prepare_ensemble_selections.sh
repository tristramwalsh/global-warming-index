#!/bin/bash
#
# Builds run-gwi.sh's list of gwi.py runs from its member settings
# (ERF_MEMBERS and GMT_MEMBERS). run-gwi.sh sources this file. It only
# decides which members each gwi.py run is given; the sampling of ensemble
# members inside gwi.py is separate, and unaffected.
#
# build_member_runs ERF_MEMBERS GMT_MEMBERS
#   Turns the two member settings into the gwi.py runs to make, one per
#   (ERF member, GMT member) pair, held in two lists of the same length:
#     ERF_LIST  the ERF member of each run, e.g. (all all all)
#     GMT_LIST  the GMT member of each run, e.g. (1 2 3)
#   Run n uses ERF_LIST[n] with GMT_LIST[n]. Where one side is all, its list
#   repeats all once per run, so that the two lists line up run by run.
#   Also sets MEMBER_TAG, the selection as a filename-safe tag for job and
#   log names, e.g. ERF-all_GMT-1-3.
#   If the combination isn't allowed, prints why and returns 1.
#
# The cases are described next to ERF_MEMBERS and GMT_MEMBERS in run-gwi.sh.

# The two pairing keywords
PAIR_WITH_GMT=paired_with_GMT_member
PAIR_WITH_ERF=paired_with_ERF_member

build_member_runs() {
  local erf_members="$1"
  local gmt_members="$2"
  local member

  # Start with no runs
  ERF_LIST=()
  GMT_LIST=()

  # ERROR: a pairing keyword on the wrong side
  if [ "${erf_members}" = "${PAIR_WITH_ERF}" ] || \
     [ "${gmt_members}" = "${PAIR_WITH_GMT}" ]; then
    echo "Pairing keyword on the wrong side: use" \
         "ERF_MEMBERS=${PAIR_WITH_GMT} or GMT_MEMBERS=${PAIR_WITH_ERF}." >&2
    return 1
  fi

  # ERROR: pairing keywords on both sides, so no members to pair
  if [ "${erf_members}" = "${PAIR_WITH_GMT}" ] && \
     [ "${gmt_members}" = "${PAIR_WITH_ERF}" ]; then
    echo "ERF_MEMBERS and GMT_MEMBERS cannot both be pairing keywords: list" \
         "the members on one side." >&2
    return 1
  fi

  # Members are listed like {1..50}. Brace expansion does NOT happen when a
  # setting is assigned, so the settings hold that literal text; eval applies
  # the expansion below: {1..3} -> 1 2 3 ; 7 -> 7.

  # CASE 1: all and all: a single run, sampling from both whole ensembles
  if [ "${erf_members}" = "all" ] && [ "${gmt_members}" = "all" ]; then
    ERF_LIST=(all)
    GMT_LIST=(all)

  # CASE 2a: ERF members listed, GMT all: one run per ERF member, each
  # sampling from the whole GMT ensemble
  elif [ "${gmt_members}" = "all" ] && \
       [ "${erf_members}" != "${PAIR_WITH_GMT}" ]; then
    for member in $(eval echo "${erf_members}"); do
      ERF_LIST+=("${member}")
      GMT_LIST+=(all)
    done

  # CASE 2b: GMT members listed, ERF all: one run per GMT member, each
  # sampling from the whole ERF ensemble
  elif [ "${erf_members}" = "all" ] && \
       [ "${gmt_members}" != "${PAIR_WITH_ERF}" ]; then
    for member in $(eval echo "${gmt_members}"); do
      ERF_LIST+=(all)
      GMT_LIST+=("${member}")
    done

  # CASE 3a: ERF paired with the GMT members listed: one run per member,
  # with the same member on both sides
  elif [ "${erf_members}" = "${PAIR_WITH_GMT}" ] && \
       [ "${gmt_members}" != "all" ]; then
    for member in $(eval echo "${gmt_members}"); do
      ERF_LIST+=("${member}")
      GMT_LIST+=("${member}")
    done

  # CASE 3b: GMT paired with the ERF members listed: one run per member,
  # with the same member on both sides
  elif [ "${gmt_members}" = "${PAIR_WITH_ERF}" ] && \
       [ "${erf_members}" != "all" ]; then
    for member in $(eval echo "${erf_members}"); do
      ERF_LIST+=("${member}")
      GMT_LIST+=("${member}")
    done

  # ERROR: a pairing keyword with all on the other side
  elif [ "${erf_members}" = "${PAIR_WITH_GMT}" ] || \
       [ "${gmt_members}" = "${PAIR_WITH_ERF}" ]; then
    echo "Pairing needs specific members on the other side, not all." >&2
    return 1

  # ERROR: members listed on both sides, without a pairing keyword
  else
    echo "Members are listed for both ERF (${erf_members}) and GMT" \
         "(${gmt_members}). Set one side to all, or to its pairing keyword" \
         "(ERF_MEMBERS=${PAIR_WITH_GMT} or GMT_MEMBERS=${PAIR_WITH_ERF})." >&2
    return 1
  fi

  # The selection as a filename-safe tag, ERF first: e.g. ERF-all_GMT-1-50
  MEMBER_TAG="ERF-$(member_tag "${erf_members}")_GMT-$(member_tag "${gmt_members}")"
}

# One member setting as a filename-safe tag:
#   {1..50} -> 1-50 ; all -> all ; a pairing keyword -> paired
member_tag() {
  case "$1" in
    "${PAIR_WITH_GMT}"|"${PAIR_WITH_ERF}") echo paired ;;
    *) echo "$1" | tr -d '{}' | sed 's/\.\./-/g' | tr ' ' '-' ;;
  esac
}
