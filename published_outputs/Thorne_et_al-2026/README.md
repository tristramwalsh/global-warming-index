# Thorne et al. (2026): Global Warming Index outputs

GWI outputs delivered for one paper, Thorne et al. (2026), from three code versions.
Each release contains only the outputs its own code (or earlier code) made, so this
table grows release by release. Tag names give the year the calculations were run.
Outputs were added to the repository on 6 November 2026; the code at each tag is unchanged.

| Folder | Stage | Runs | Made with tag (commit) | Delivered |
| --- | --- | --- | --- | --- |
| submission_models/ | Submission; reused unchanged in the final paper | SMILE_ESM-SSP126/245/370, NorESM_rcp45-Volc/VolcConst | Thorne_et_al-2025 (c14ecfa) | 2 Apr 2025 |
| submission_observed/ | Submission | observed-SSP245: HadCRUT5.0.2 to 2024, all members in one calculation | Thorne_et_al-2025 (c14ecfa) | 2 Apr 2025 |

Not in this release, because they were made with later code:
- the submission's per-member runs on the multi-dataset observational ensemble
  to 2024: release Thorne_et_al-2025-JK (code of 20 Oct 2025);
- the revision's HadCRUT5.1 and per-member observational-ensemble runs to 2025, which
  the final paper uses: release Thorne_et_al-2026 (code of 16-17 Sep 2026).
The Thorne_et_al-2026 release contains every folder.

## Contents

results/ holds the CSV files exactly as delivered, flat; the figures sent alongside
them are not included. File names give the headline definition, scenario, ensemble
selection, variables and regression years. The files were gathered by
extract_results_Thorne-et-al.py, part of the code at each tag: it copied the
historical-only headline results, keeping only the Nat, Ant and Tot columns.

The scheduler settings of these runs were not recorded at the time, so no configs are
included here. The schedule-gwi.sh in the code at each tag holds only its last-used
settings, not a record of every run.

To rerun, check out the tag that made a folder and run from the repository root. These
files are a record of what was run, not kept compatible with later code.

Outputs: CC BY 4.0. Code: MIT.
