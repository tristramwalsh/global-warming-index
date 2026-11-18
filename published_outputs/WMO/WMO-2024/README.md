# WMO State of the Global Climate 2024

GWI outputs for the preliminary assessment of human-induced warming in the WMO State of the Global Climate 2024. WMO SotC is produced early in the year before all input datasets can be finalised; the warming attribution values therefore were preliminary: the IGCC 2023 attribution results were extrapolated by one year at each attribution method's own current warming rate; the extrapolated results were combined as in the IGCC multi-method assessment approach. A full GWI analyses run with preliminary 2024 data checked that the extrapolation was reasonable (it was); the 2024 run was not used in the published estimate, for consistency with the other methods.

## Publication and code

| Type | Name | DOI |
| --- | --- | --- |
| Publication | WMO State of the Global Climate 2024, https://wmo.int/files/state-of-global-climate-2024 | — |
| GWI code | Tag `IGCC-2023` (commit `618350a`): the 2023 results that were extrapolated | [10.5281/zenodo.24022666](https://doi.org/10.5281/zenodo.24022666) |
| GWI code | Tag `WMO-2024` (commit `be26172`): the 2024 validation run | [10.5281/zenodo.24022833](https://doi.org/10.5281/zenodo.24022833) |

## Contents

| Scenario | Description | Results | Slurm Config |
| --- | --- | --- | --- |
| `observed-2023` | HadCRUT5.0.2.0 observations to 2023 with the IGCC 2023 ERFs; identical to the IGCC-2023 outputs | ✅ | ❌ |
| — | The 2023 GWI extrapolated to 2024 with the 2014-2023 warming rate (`Extrapolation_2023-to-2024.csv`); used in the published estimate | ✅ | ❌ |
| `observed-2024` | HadCRUT5.0.2.0 observations to 2024 with preliminary 2024 ERF update; validation only | ❌ | ❌ |

✅ included · ❌ not included

## License

Outputs: CC BY 4.0. Cite the GWI code version and the publication.
