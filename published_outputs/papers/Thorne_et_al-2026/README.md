# Thorne et al. (2026): Global Warming Index outputs

All GWI outputs delivered for Thorne et al. (2026).
- Tag names here give the year the calculations were run (the timeline for producing the paper was long).
- Outputs were added to the repository on 6 November 2026; the code at each tag is unchanged.
- To rerun, check out the tag that made a folder and run from the repository root. Scheduler settings were kept only for the revision runs (see Contents).
- These files are a record of what was run, not kept compatible with later code.


## Publication and code
| Type | Name | DOI |
| --- | --- | --- |
| Publication | Thorne et al. (2026) (preprint; final DOI to follow) | [10.5194/essd-2025-825](https://doi.org/10.5194/essd-2025-825) |
| Paper's archive | Nicklas (2026) | [10.5281/zenodo.23568234](https://doi.org/10.5281/zenodo.23568234) |
| GWI code | `Thorne_et_al-2025` | [10.5281/zenodo.24022860](https://doi.org/10.5281/zenodo.24022860) |
| GWI code | `Thorne_et_al-2025-JK` | [10.5281/zenodo.24023089](https://doi.org/10.5281/zenodo.24023089) |
| GWI code | `Thorne_et_al-2026`: contains every output in this folder | [10.5281/zenodo.24023281](https://doi.org/10.5281/zenodo.24023281) |


## Contents

- The final paper uses `submission_models/` and `revision_observed/`
- All scenarios output four post-processing time-periods:
    - `ANNUAL`: annual mean current year
    - `SR1.5`: SR1.5 definition is 30 year mean centred on the current year; future half of mean is given by constrained attributed extrapolation at current warming rate
    - `AR6`: AR6 definition is lagged decade average, with the current year as the last year of the decade
    - `CWGL`: GWI-CGWL is 20-year mean centred on current year; future half of mean is given by constrained attributed projections following SSPs. This is an experimental hybrid of the GWI and CWGL (https://doi.org/10.1038/d41586-023-03775-z) methodologies
- `extract_results_Thorne-et-al.py` was edited on a case by case basis to gather the results files from the runs.


| Folder | Scenario | Description | Made with, delivered | Results | Config |
| --- | --- | --- | --- | --- | --- |
| `submission_models/` | `SMILE_ESM-SSP126`, `-SSP245`, `-SSP370` | MPI-ESM1.2-LR large ensemble, GWI attribution run against each member separately, and all members; reused unchanged in the final paper | `Thorne_et_al-2025` (`c14ecfa`), 2 Apr 2025 | ✅ | ❌ |
| `submission_models/` | `NorESM_rcp45-Volc`, `NorESM_rcp45-VolcConst` | NorESM large ensembles with and without future volcanic eruptions; GWI attribution run against each member separately; reused unchanged in the final paper | `Thorne_et_al-2025` (`c14ecfa`), 2 Apr 2025 | ✅ | ❌ |
| `submission_observed/` | `observed-SSP245` | HadCRUT5.0.2.0 to 2024, all members in one calculation; submission only | `Thorne_et_al-2025` (`c14ecfa`), 2 Apr 2025 | ✅ | ❌ |
| `submission_observed/` | `observed_JK-2024-SSP245` | Multi-dataset observational ensemble to 2024, members 0-99 attributed separately; superseded by the revision | `Thorne_et_al-2025-JK` (`c414146`), 20 Oct 2025 | ✅ | ❌ |
| `revision_observed/` | `observed-2025-SSP245` | HadCRUT5.1.0.0 to 2025, all members in one calculation; used in the final paper | `Thorne_et_al-2026`, 17 Sep 2026 | ✅ | ✅ `configs/schedule-gwi_aa8f499_HadCRUT5.1.sh` |
| `revision_observed/` | `observed_JK-2025-SSP245` | Multi-dataset observational ensemble to 2025, members 1-100 attributed separately; used in the final paper | `Thorne_et_al-2026`, 17 Sep 2026 | ✅ | ✅ `configs/schedule-gwi_9b44f54_JK-2025.sh` |



## License
Outputs: CC BY 4.0. Code: MIT.
