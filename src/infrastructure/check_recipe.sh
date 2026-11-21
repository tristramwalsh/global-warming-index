#!/bin/bash
#
# Checks a recipe against the template before run-gwi.sh reads it.
# run-gwi.sh sources this file.
#
# check_recipe RECIPE TEMPLATE
#   The recipe must set exactly the settings the template sets, each once,
#   and nothing else. This catches a typo (which would otherwise set a
#   variable nothing reads), a setting missed when one is added to the
#   template, and a stray variable that would overwrite one of run-gwi.sh's
#   own. The names are read from the file text, before the recipe is
#   sourced, so nothing can be filled in by a variable that happens to be set
#   in the shell.
#   If the recipe doesn't match, prints what is missing, unknown or repeated,
#   and returns 1.

# The names a file assigns at the start of a line, e.g. RECIPE_VARS from
# RECIPE_VARS=GHG,OHF,Nat, one per line and sorted.
assigned_names() {
  grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$1" | sed 's/=$//' | LC_ALL=C sort
}

check_recipe() {
  local recipe="$1"
  local template="$2"
  local template_names recipe_names missing unknown repeated

  template_names=$(assigned_names "${template}")
  recipe_names=$(assigned_names "${recipe}")
  # comm compares two sorted lists: -23 keeps the names only in the first,
  # -13 those only in the second
  missing=$(LC_ALL=C comm -23 <(echo "${template_names}") \
                              <(echo "${recipe_names}" | uniq) | sed '/^$/d')
  unknown=$(LC_ALL=C comm -13 <(echo "${template_names}") \
                              <(echo "${recipe_names}" | uniq) | sed '/^$/d')
  repeated=$(echo "${recipe_names}" | uniq -d)

  if [ -n "${missing}${unknown}${repeated}" ]; then
    echo "The recipe ${recipe} doesn't match ${template#./}," \
         "which lists every setting:" >&2
    # Unquoted on purpose, to print each list on one line
    [ -n "${missing}" ]  && echo "  missing: " ${missing} >&2
    [ -n "${unknown}" ]  && echo "  unknown: " ${unknown} >&2
    [ -n "${repeated}" ] && echo "  repeated:" ${repeated} >&2
    echo "A recipe sets every setting in the template, once each, and" \
         "nothing else. Copy any missing settings from the template." >&2
    return 1
  fi
}
