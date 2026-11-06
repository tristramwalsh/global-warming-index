# Thorne et al. (2026): Global Warming Index outputs

All GWI outputs delivered for one paper, Thorne et al. (2026), from three code versions.
Tag names give the year the calculations were run. Outputs were added to the repository
on 6 November 2026; the code at each tag is unchanged.

| Folder | Stage | Runs | Made with tag (commit) | Delivered |
| --- | --- | --- | --- | --- |
| submission_models/ | Submission; reused unchanged in the final paper | SMILE_ESM-SSP126/245/370, NorESM_rcp45-Volc/VolcConst | Thorne_et_al-2025 (c14ecfa) | 2 Apr 2025 |
| submission_observed/ | Submission | observed-SSP245: HadCRUT5.0.2 to 2024, all members in one calculation | Thorne_et_al-2025 (c14ecfa) | 2 Apr 2025 |
| submission_observed/ | Submission; superseded in the final paper by the revision | observed_JK-2024-SSP245: multi-dataset observational ensemble (Kennedy and Calvert) to 2024, each of members 0-99 attributed separately | Thorne_et_al-2025-JK (c414146) | 20 Oct 2025 |
| revision_observed/ | Revision; used in the final paper | observed-2025-SSP245: HadCRUT5.1 to 2025, all members in one calculation; observed_JK-2025-SSP245: updated observational ensemble to 2025, each of members 1-100 attributed separately | Thorne_et_al-2026 (launched from aa8f499; contains 7c75b18) | 17 Sep 2026 |

The final paper uses submission_models/ and revision_observed/.

## Contents

results/ holds the CSV files exactly as delivered, flat; the figures sent alongside
them are not included. File names give the headline definition, scenario, ensemble
selection, variables and regression years. The files were gathered by
extract_results_Thorne-et-al.py, part of the code at each tag: it copied the
historical-only headline results, keeping only the Nat, Ant and Tot columns.

revision_observed/configs/ holds the two scheduler settings committed as records of the
revision runs: aa8f499 for the HadCRUT5.1 run, and 9b44f54 for the observational-ensemble
run, submitted in two halves (members 1-50, then 51-100) that differed only in the member
range. The submission runs' settings were not recorded at the time, so no configs are
included for them; the schedule-gwi.sh in the code at each tag holds only its last-used
settings.

The revision jobs were launched from aa8f499 (16 Sep 2026) and ran while performance,
memory and logging changes were committed up to 7c75b18 (17 Sep 2026), which the
Thorne_et_al-2026 tag contains. None of those changes was intended to alter results.

To rerun, check out the tag that made a folder and run from the repository root. These
files are a record of what was run, not kept compatible with later code.

Outputs: CC BY 4.0. Code: MIT.
