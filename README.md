# The Global Warming Index

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24020859.svg)](https://doi.org/10.5281/zenodo.24020859)
[![Code licence: MIT](https://img.shields.io/badge/code%20licence-MIT-yellow.svg)](LICENSE)
[![Outputs licence: CC BY 4.0](https://img.shields.io/badge/outputs%20licence-CC%20BY%204.0-lightgrey.svg)](published_outputs/LICENSE.md)

## Introduction

The Global Warming Index (GWI) is a method for attributing observed global temperature changes to different climate forcing components.

It uses a two-step regression approach: first, prior estimates of the temperature responses to individual forcings (greenhouse gases, other human forcing, and natural forcings) are generated using a climate emulator model, then these modelled responses are constrained via regression against observed temperatures and pre-industrial control simulations to determine posterior estimate.

The pipeline is designed to be is fully modular and flexible for our quick turnaround operational attribution of arbitrary scenarios and data. Modularity currently covers the input data, climate model, reference temperatures, reference constraining period, preindustrial baseline, targeting individual ensemble members, headline warming definition (annual mean, SR1.5, AR6, CGWL, etc), rate calculation, warming component outputs, etc. Further modularity will soon also include turning on and off individual uncertainty sources, selecting regression method to better handle differences between historical-information-only or full-information constraining, rate definitions, and adding additional attributable natural components.

Very large Monte Carlo simulations (with $\mathcal{O}(10^8)$ ensemble members) are used to produce probabilistic estimates of human-caused warming, natural contributions, and total warming with full uncertainty quantification across multiple sources of uncertainty, including:
- radiative forcing uncertainty
- climate parameter uncertainty
- observed temperature uncertainty
- possible alternative realisations of internal variability

## Results

The Global Warming Index is hosted by the Environmental Change Institute (ECI) at the University of Oxford, and is updated operationally as part of the Indicators of Global Climate Change (IGCC) project.

Every set of GWI results that has been published is in [`published_outputs/`](published_outputs/), with a README for each saying where it was published and which release made it:

- `published_outputs/IGCC/`: the annual IGCC updates.
- `published_outputs/WMO/`: contributions to the WMO State of the Global Climate.
- `published_outputs/papers/`: results provided for individual papers.

### IGCC outputs
The annual IGCC global assessment, which combines the GWI with the other attribution methods is at:
- GitHub Code: https://github.com/ClimateIndicator/anthropogenic-warming-assessment
- Zenodo Results: https://doi.org/10.5281/zenodo.7883757

## How to cite

If you use GWI code or results, cite the release you used and the method papers:


### Citing the GWI *generally*
To refer to the operational Global Warming Index (GWI) project code in general rather than a particular release, use the
- **Project concept DOI (representing all operational updates)**
    - Walsh, T. Global Warming Index. Zenodo. https://doi.org/10.5281/zenodo.24020859
- **Most recent GWI-specific methodology paper, providing uncertainty quantification**
    - Haustein, K., Allen, M.R., Forster, P.M. et al. A real-time Global Warming Index. Sci Rep 7, 15417 (2017). https://doi.org/10.1038/s41598-017-14828-5
- **Concept paper proposing the GWI for policy use**
    - Otto, F., Frame, D., Otto, A. et al. Embracing uncertainty in climate change policy. Nature Clim Change 5, 917–920 (2015). https://doi.org/10.1038/nclimate2716


### Using and citing the *most recent* published GWI data release (2025)
- **GWI Project, 2025 release DOI**
    - Walsh, T. (2026). Global Warming Index (Version IGCC-2025) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24023147
- **IGCC Publication, 2025 global indicators journal article**
    - Forster, P. M., Walsh, T., Smith, C., et al.: Indicators of Global Climate Change 2025: annual update of key indicators of the state of the climate system and human influence, Earth Syst. Sci. Data, 18, 3889–3933, https://doi.org/10.5194/essd-18-3889-2026, 2026.
- **Most recent methods paper**
    - Haustein, K., Allen, M.R., Forster, P.M. et al. A real-time Global Warming Index. Sci Rep 7, 15417 (2017). https://doi.org/10.1038/s41598-017-14828-5

### Using and citing other specific GWI project outputs

If you use results from `published_outputs/`, e.g. the outputs for other specific papers, please follow the citations given in that folder's README, which covers both published outputs and specific code release DOIs.

## Releases

Each release has a git tag and GitHub release, and is archived on Zenodo as a version named after the tag, under the concept DOI https://doi.org/10.5281/zenodo.24020859 (each version's own DOI is on its Zenodo page).

The code is MIT-licensed ([`LICENSE`](LICENSE)); the published outputs are [`CC BY 4.0`](published_outputs/LICENSE.md); the input data keep their own licences (see the `SOURCE.md` in each data folder).

Changes between releases are in [`CHANGELOG.md`](CHANGELOG.md), and the release process is in [`RELEASING.md`](RELEASING.md).

| Release | Made for | Outputs in `published_outputs/` |
| --- | --- | --- |
| `Thorne_et_al-2026` | Thorne et al. (2026), revision, [10.5194/essd-2025-825](https://doi.org/10.5194/essd-2025-825) | `papers/Thorne_et_al-2026/` (all three Thorne releases) |
| `IGCC-2025` | IGCC 2025: Forster et al. (2026), [10.5194/essd-18-3889-2026](https://doi.org/10.5194/essd-18-3889-2026) | `IGCC/IGCC-2025/` |
| `IGCC-2024` | IGCC 2024: Forster et al. (2025), [10.5194/essd-17-2641-2025](https://doi.org/10.5194/essd-17-2641-2025) | `IGCC/IGCC-2024/` |
| `WMO-2024` | WMO State of the Global Climate 2024, https://wmo.int/files/state-of-global-climate-2024 | `WMO/WMO-2024/` |
| `IGCC-2023` | IGCC 2023: Forster et al. (2024), [10.5194/essd-16-2625-2024](https://doi.org/10.5194/essd-16-2625-2024) | `IGCC/IGCC-2023/` |
| `IGCC-2022` | IGCC 2022: Forster et al. (2023), [10.5194/essd-15-2295-2023](https://doi.org/10.5194/essd-15-2295-2023) | `IGCC/IGCC-2022/` |
| none (a spreadsheet that predates this code) | Otto et al. (2015), [10.1038/nclimate2716](https://doi.org/10.1038/nclimate2716) | `papers/Otto_et_al-2015/` |



## Replicating the Results
Prerequisites:
- **Package Manager:** `Micromamba` (Conda may not solve dependencies for all systems)
- **HPC:** `SLURM` for full-size runs. Small runs, such as quick tests and development work, can also run directly on a laptop or an interactive node (see **Run modes**).

### 1. Setting up the environment
1. Clone the repository:
   ```bash
   git clone https://github.com/tristramwalsh/global-warming-index.git
   cd global-warming-index
   ```

2. Create a new environment with the required dependencies (tested on Oxford's ARC and OUCE Linux systems):
   ```bash
   micromamba env create -f environment.yml
   micromamba activate gwi
   ```

### 2. Running the Analysis

The workflow is controlled by the `run-gwi.sh` script, which calculates the GWI for every configuration set at its top: either by submitting a SLURM job for each, or by running them directly in the current shell.

#### 2.1. Basic Usage

##### 2.1.1. Run the GWI iterations
Edit the settings at the top of `run-gwi.sh` (see **Configuration Arguments** below), activate the `gwi` environment, then:
```bash
bash run-gwi.sh
```
The script checks that the `gwi` environment is active before running anything; it does not activate it for you, as that differs between machines.

**Run modes** (`MODE`):
- `slurm` = generate and submit one SLURM job per configuration. Use this for full-size runs.
- `direct` = run each configuration now, one after another, in this shell: on a laptop or on an interactive node. Use this for quick tests and development. It stops at the first failure, and logs go to `logs_direct/` (or `logs_direct_<tag>/`).
- `dry` = print the `gwi.py` commands that would run, and run nothing.

For a quick test, set `MODE=direct` and uncomment the **Quick test** settings in `run-gwi.sh`; on a laptop this takes well under a minute.

##### 2.1.2. Combine results from multiple iterations
Once the runs have completed, run the combination script with the same tag:

```bash
python combine_results_iterations.py
```

This averages the repeat iterations, where applicable derives the historical-information-only timeseries (when a sequence of regression end years are used), and produces all associated figures.

See **Combining Results from Multiple Iterations** for its options, and **Output Structure** for where everything is written.

#### 2.2. Configuration Arguments

The following parameters in `run-gwi.sh` control the analysis:

**`MODE`**: How to run each configuration: `slurm`, `direct` or `dry` (see **Run modes** above).

**`START_REGRESS`**: The starting year for the observational constraint period (e.g., `1850`)
- Defines when start year of the data ranges that the regression constraint is applied over.


**`END_REGRESS`**: The ending year(s) for observational constraint period
- Full-information GWI: For a single end year: just specify one value
    - e.g. `2023` creates a single job with regression from 1850-2023
- Historical-only GWI: To generate GWI so that the GWI in year Y is based only on information up to and including year Y, you need to repeatedly re-run the GWI analysis with the regression constraint ending in every year of the historical-only timeseries.
    - e.g. `seq 1950 2024` creates separate jobs for each year from 1950-2024, each only incorporating years from START_REGRESS to END_REGRESS in the constraining.


**`SUBSAMPLE_ITERATIONS`**: The ensemble subsampling sizes
- Assuming independence, the full ensemble contains $\mathcal{O}(10^8)$ samples, which is computationally very heavy (memory). To reduce this to a size manageable for computation, the code randomly samples a subset of ensemble members from each source (observational uncertainty, forcing uncertainty, climate parameter uncertainty, internal variability uncertainty).
- This argument specifies how many samples to take from each source for each iteration (each value in the array), and how many iterations to run (the length of the array)
- When repeat runs are generated, the results are combined using a weighted average (weighted by ensemble size of each iteration) in `combine_results_iterations.py`, to minimise sampling noise.
- Examples:
    - e.g.`(90 90 90 90)` runs 4 repeat iterations with a maximum of 90 samples for each of the 4 uncertainty sources (observations, forcing, climate parameters, internal variability)
    - e.g. `(30 60 90)` runs 3 repeat iterations with 30, 60, and 90 samples respectively

**`SEED`**: A seed for the random sampling of ensemble members, for exactly reproducible runs
- Leave empty (the usual setting) for unseeded sampling: every run samples afresh, so repeat iterations differ, and averaging them reduces the sampling noise.
- Set to a whole number (e.g. `42`) to draw exactly the same samples each time: runs of the same configuration on the same inputs then give identical results. This is useful for checking that a code change leaves the results unchanged.
- Every run uses the same seed, so with a seed set, repeat iterations of the same configuration in `SUBSAMPLE_ITERATIONS` give identical results.


**`PREINDUSTRIAL_ERA`**: Pre-industrial baseline period
- Used to define the baseline pre-industrial proxy for temperatures to be produced relative to.
- Examples:
    - `1850-1900` (default) = 1850-1900 average as pre-industrial baseline
    - `1981-2010` = 1981-2010 average as pre-industrial baseline
    - `n` = no pre-industrial offset applied (temperatures are left in their default state, which may be relative to a different baseline depending on the dataset)

**`INCLUDE_REG_CONST`**: Toggle regression constant (`y` or `n`)
- Specifies whether to include a constant term offset within the multi-variable regression. We typically set this to `n`, as in `run-gwi.sh`. (`gwi.py` itself defaults to `y` if the option is not passed.)
- Examples:
    - `y` = include the constant term in the regression
    - `n` = do not include the constant term in the regression

**`VARS`**: Variables to regress on (comma-separated, no spaces)
- `GHG,OHF,Nat` = 3-way regression (greenhouse gases, other human forcings, natural)
- `Ant,Nat` = 2-way regression (anthropogenic total, natural)
- `Tot` = 1-way regression (total forcing only)

**`INCLUDE_SUB_VARS`**: Include sub-variables in output (`y` or `n`)
- Calculates component-wise contributions to the aggregate variables (e.g. GHG = CO2 + CH4 + N2O + F-gases).
- Useful for detailed attribution analysis while maintaining robust regression on aggregate variables.
- Note that these components must be included in the forcing dataset individually for this to be possible.


**`ERF_DATASET`** and **`GMT_DATASET`**: Which forcing (ERF) and reference temperature (GMT) datasets to attribute against each other, one of each per run
- Datasets are named by ID: `<product or experiment>[-<year>][-<variant>]`. The year, for observed datasets only, is the year the dataset's history ends; observed temperatures are cut off at that year.
- Observed ERF datasets:
    - `IGCC-<YYYY>` = IGCC historical ERFs, 1000 members.
        - `<YYYY>` = `2022`, `2023`, `2024`, `2025`
        - Note: `INCLUDE_SUB_VARS=y` needs `2024` or `2025`.
    - `SSP_extension-2022-<SSP>` = observed history to 2022, then `<SSP>` to 2100, 841 members.
        - `<SSP>` = `SSP119`, `SSP126`, `SSP245`, `SSP370`, `SSP434`, `SSP460`, `SSP534-over`, `SSP585`.
        - Useful: (i) to produce constrained future projections, and (ii) to calculate GWI-CWGL hybrid headlines, which require those future projections to be available.
        - Note: The historical ERFs are generated differently from the IGCC ERFs, so the two differ before 2022 too.
- Observed GMT datasets:
    - `HadCRUT-<YYYY>` = HadCRUT5 ensemble, 200 members.
        - `<YYYY>` = `2022`, `2023`, `2024`, `2025`.
    - `JK_multi-<YYYY>` = John Kennedy's multi-dataset ensemble, 100 members.
        - `<YYYY>` = `2024`, `2025`.
- Simulated datasets (climate model large ensembles) provide both an ERF and a GMT dataset, and pair only with themselves: set `ERF_DATASET` and `GMT_DATASET` to the same ID.
    - `SMILE_ESM-<SSP>` = MPI-ESM1.2-LR: historical to 2014
        - `<SSP>` = `SSP126`, `SSP245`, `SSP370`.
        - A single ERF timeseries, and 50 GMT members.
    - `FutureVolc_ESM-rcp45-<Volcanism>` = NorESM1-M (Bethke et al., 2017): historical to 2005, then RCP4.5 with.
        - `<Volcanism>` = `Volc` or `VolcConst`
            - `Volc` = stochastic future volcanic eruptions
                - A single Anthropogenic ERF timeseries, and 60 Volcanism scenarios, each with a different volcanic forcing, with a 1-1 pairing between ERF and GMT members
            - `VolcConst` = or constant background volcanic forcings
                - A single ERF timeseries, and 60 GMT members, with no direct pairing needed between ERF and GMT members.
- Any observed ERF dataset can be paired with any observed GMT dataset, as long as both cover the regressed years.
- e.g. `ERF_DATASET=IGCC-2025` with `GMT_DATASET=HadCRUT-2025`, for the IGCC assessment.


**`ERF_MEMBERS`** and **`GMT_MEMBERS`**: Which ensemble members of each dataset to use
- `all` = sample from the whole ensemble (the usual setting)
- `7` = a single member, by its label in the dataset
- `{1..100}` = each member in turn, one `gwi.py` run each, sampling from the whole ensemble on the other side (set to `all`)
- `paired_with_GMT_member` (for `ERF_MEMBERS`) or `paired_with_ERF_member` (for `GMT_MEMBERS`) = one run per member listed on the other side, with the ERF and GMT members of the same label. Required for `FutureVolc_ESM-rcp45-Volc`, which has one natural ERF member for each GMT member.
- Members listed on both sides without a pairing keyword, pairing keywords on both sides, or a pairing keyword opposite `all`, stop the script with an error before anything runs.
- Member labels are each dataset's own: HadCRUT `{1..200}`, JK_multi `{1..100}`, SMILE_ESM GMT `{0..49}`, FutureVolc_ESM `{1..60}`. IGCC and SSP_extension ERF members are labelled by configuration number, so use `all` for them.
- See **Running the GWI for Individual Ensemble Members** for running the full calculation separately for each member.


**`PICONTROL_DATASET`**: The piControl simulations that sample internal variability
- `CMIP6` = the CMIP6 piControl simulations in `data/piControl/GMT/CMIP6/` (the only one for now)


**`COMMITTED`**: Calculate committed warming (`n` or `start-end`)
- Calculates the warming committed to occur by a future year if effective radiative forcing (ERF) is held constant from a specified start year.
- Format: `start_year-end_year`
- You can use `end_regress` as a keyword for the start year.
- Examples:
    - `n`: No committed warming calculation (default)
    - `2024-2300`: Hold ERF constant from 2024 to 2300
    - `end_regress-2300`: Hold ERF constant from the end of the regression period to 2300
- The results folder name gains `_COMMITTED--<start>-<end>`.
- Ensure `TRUNCATION` extends to at least the committed end year.
- **Note**: The regression end year (`END_REGRESS`) cannot be later than the start of the committed warming period (otherwise the sections of the ERF timeseries and reference temperatures being regressed against each other will no longer correspond with each other). This mismatch will raise a `ValueError` to prevent incorrect calculations.


**`TRUNCATION`**: Year range for output timeseries (e.g., `1850-2050`)
- Output is cut down to this range after regression, but BEFORE the headline calculations.
- This enables constrained future projections to be produced if future ERF scenario forcings are available; in this case, the truncation will extend beyond the regression range.
- NOTE: Where CGWL headlines are required, the truncation range must extend far enough into the future to accommodate the 10-year future averaging period (see Important Technical Details below).


**`INCLUDE_RATE`**: Calculate warming rates (`y` or `n`)
- Definition of warming rate is the linear trend over the previous 10 years, and is (currently) hard coded to produce rates from 1950 to the end of the truncation range.
- Rates are very (very) computationally expensive (time); set to `n` unless specifically needed.


**`HEADLINE_TOGGLES`**: Which headline definitions to calculate
- `'annual,SR1.5,AR6,CGWL'` = all four definitions
- `n` = skip headline calculations (only produce annual timeseries)


**`HEADLINE_YEARS`**: Which years to calculate headlines for
- `'end_regress'` = use the end of regression range (recommended for historical-only)
- `'end_trunc'` = use the end of truncation range
- `'IGCC'` for latest year, 2017 repeat, and 2010-2019 repeat (for IPCC validation)
- `'2024'` = specific single year
- `'2023,2024,2025'` = multiple specific years
- `'end_regress,2050,2100,2300'` to combine end_regress and manual years
- `$(seq -s, 1950 2024)` = all years in a range (expensive)


**`CALCULATE_PRIORS_OUTPUT`**: Also output the prior (pre-constraint) warming (`y` or `n`)
- The priors depend only on the ERF ensemble and FaIR parameters, not on the reference temperatures. These are useful for understanding the prior distribution of warming before the observational constraint. All ensemble members are used for this, so note that, depending on the size of the GWI Monte Carlo ensemble, the output priors could be a different sample than used in the GWI calculation itself. The priors are saved in the same hierarchy as the GWI results, but in `results/priors/` (or `results_<tag>/priors/`).
- `y` = calculate and save the priors (default in `gwi.py`)
- `n` = skip them, e.g. for member-by-member runs that select only temperature members, where they would be identical for every member


**`OUTPUT_TAG`**: A name for this piece of work (e.g. `IGCC2025`)
- Keeps separate pieces of work apart, so that they neither overwrite each other nor all get processed together: results go to `results_<tag>/`, figures to `plots_<tag>/` and logs to `logs_slurm_<tag>/` (or `logs_direct_<tag>/` in direct mode).
- Pass the same tag to `combine_results_iterations.py` (`--output-tag=<tag>`).
- Left empty, the plain `results/`, `plots/` and `logs_slurm/` (or `logs_direct/`) folders are used.


**Slurm resources** (slurm mode only): `CLUSTER`, `SIM_CPUS`, `CPU_MEM` and `WALLTIME`
- `CLUSTER` = which Slurm cluster to submit to on Oxford ARC: `arc`, `htc`, or `arc,htc` to take whichever starts sooner. The partition is set automatically from the host.
- `SIM_CPUS`, `CPU_MEM`, `WALLTIME` = CPUs, memory per CPU, and time limit for each job.
- Our standard runs use `SIM_CPUS=28`, `CPU_MEM=9000M` and `WALLTIME=0:10:00`, which suits `SUBSAMPLE_ITERATIONS` of about 60-100 samples without rates. Larger ensembles need more memory (see **Memory and Computing Resources**), and rates or member-by-member runs need more time.

#### 2.3. What the Script Does

The `run-gwi.sh` script:
1. Loops over `END_REGRESS` values (regression end years) and `SUBSAMPLE_ITERATIONS` (subsampling sizes)
2. For each combination, calls `gwi.py` with the specified CLI arguments (once per selected member, one after another, when `ERF_MEMBERS` or `GMT_MEMBERS` list members; the list of runs is built in `src/prepare_ensemble_selections.sh`):
    - in slurm mode, from a SLURM job named after the run's configuration, which it submits; each job's log goes to `logs_slurm/` (or `logs_slurm_<tag>/`)
    - in direct mode, immediately in the current shell; each combination's log goes to `logs_direct/` (or `logs_direct_<tag>/`)


#### 2.4. Technical Notes

##### 2.4.1. Truncation vs Regression Ranges

The truncation range defines the output timeseries extent, while the regression range determines which years are used to fit the scaling coefficients that constrain the prior temperature response.

- **Regression range** (`--regress-range`): Years used to calculate the regression coefficients between modeled responses and observations
  - Must be within both the observed temperature data range AND the forcing data range
  - If it isn't, the run stops with an error, rather than shortening the range
  
- **Truncation range** (`--truncate`): Years included in the final output timeseries
  - Can extend beyond the regression range (e.g., for generating constrained future projections)
  - Must be within the forcing data range
  - The code automatically adjusts if specified outside available forcing data

**Important for headline calculations**: 
- Headline definitions (SR1.5, AR6, CGWL) require specific year ranges to be available in the truncation period:
  - **SR1.5**: Needs 15 years before and including the target year (30-year centered mean)
  - **AR6**: Needs 9 years before and including the target year (10-year lagged mean)
  - **CGWL**: Needs 9 years before and 10 years after the target year (20-year centered mean)
  
If you set `HEADLINE_YEARS='end_regress'`, make sure your truncation range extends far enough to accommodate these definitions. For example, if `end_regress=2024` and you want CGWL, you need `end_trunc >= 2034`, and forcing projections that extend that far.

##### 2.4.2. Variable Regression Logic

The code handles different regression configurations automatically.

- **3-way regression** (`GHG,OHF,Nat`): Regresses three components separately. Automatically also calculates: 
    1. `Ant` (Anthropogenic) = GHG + OHF
    2. `Tot` (Total) = GHG + OHF + Nat
    3. `Res` (Residual) = Observed - Tot


- **2-way regression** (`Ant,Nat`): Regresses total anthropogenic and natural separately. Automatically also calculates:
    1. `Tot` = Ant + Nat
    2. `Res` = Observed - Tot

- **1-way regression** (`Tot`): Regresses total forcing only. Automatically also calculates:
    1. `Res` = Observed - Tot


##### 2.4.3. Combining Results from Multiple Iterations
After running multiple iterations, use `combine_results_iterations.py` to aggregate the results:

```shell
python combine_results_iterations.py --output-tag=<tag> --re-calculate=<y/n> --plot-every=<N>
```

All three options are optional:
- `--output-tag=<tag>` = the `OUTPUT_TAG` used in `run-gwi.sh`, so that the script reads and writes that piece of work's folders. Default: none (the plain `results/` and `plots/`).
- `--re-calculate=y|n` = `y` (default) recomputes the averages of the iterations, and the historical-only timeseries derived from them, and saves both to the aggregated folder. `n` reads both back from the aggregated folder, as written by an earlier `y` run, and only draws the figures.
- `--plot-every=N` = thins the figures of member-by-member selections (see **Running the GWI for Individual Ensemble Members**). Default `10`; `1` draws every figure. The standard `ERF-all_GMT-all` selection is always plotted in full.

###### 2.4.3.1 Aggregating Iteration Outputs
For iterations generated with `SUBSAMPLE_ITERATIONS`, the script:
1. Finds all iterations generated from `SUBSAMPLE_ITERATIONS` in `results/iterations/`
2. Combines them using a weighted average (weighted by ensemble size)
3. Saves combined results to `results/aggregated/`

(With a tag set, these are `results_<tag>/iterations/` and `results_<tag>/aggregated/`.)

###### 2.4.3.2 Historical-Only Timeseries
For iterations generated in order to provide different regression end years (e.g., `END_REGRESS=$(seq 1950 2024)`):
1. For each regression end year (e.g., 1950, 1951, ..., 2024), a separate GWI calculation is performed
2. The `calculate_historical_only()` function extracts values for each year from the appropriate regression period
    1. When `headline` datasets are available, this pulls from the **`headlines`** dataframes at the appropriate string index
    2. When `headline` datasets are not available, this pulls from the **`timeseries`** dataframes at the index `current_year`
3. The result is a timeseries where each year's attribution is based only on data available up to that year, which provides an estimate of what the GWI would have been if calculated in real-time each year, though note that the datasets and climate parameters are still the same as used in the full-information GWI.


##### 2.4.4. Output Structure
Results are organized hierarchically (`results_<tag>/`, `plots_<tag>/` and `logs_slurm_<tag>/` or `logs_direct_<tag>/` when `OUTPUT_TAG` is set):
```
global-warming-index/
├── results/
│    └── {category}/
│        └── ERF--{erf}_GMT--{gmt}_PREINDUSTRIAL--{start}-{end}/
│            └── ENSEMBLE-MEMBER--ERF-{member}_GMT-{member}/
│                └── VARIABLES--{vars}/
│                    └── REGRESSED-YEARS--{start}-{end}/
│                        ├── GWI_results_timeseries_*.csv
│                        ├── GWI_results_headlines_*.csv
│                        └── GWI_results_rates_*.csv  (if INCLUDE_RATE=y)
├── plots/        figures drawn by combine_results_iterations.py (see below)
├── logs_slurm/   one log per Slurm job (slurm mode)
└── logs_direct/  one log per configuration (direct mode)
```

The top level names the run's inputs: the ERF and GMT dataset IDs and the pre-industrial period (`PREINDUSTRIAL--none` with no offset), plus `_COMMITTED--{start}-{end}` when `COMMITTED` is set; e.g. `ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900`. Below it, `ENSEMBLE-MEMBER--` gives the member selection, ERF first, e.g. `ERF-all_GMT-all` or `ERF-all_GMT-7`. Each file's name repeats these labels.

`{category}` is one of:
- `iterations/` = the results of each `gwi.py` run
- `aggregated/` = those averaged across iterations by `combine_results_iterations.py`, plus the historical-only timeseries
- `observations/`, `erfs/`, `priors/` = the observed temperatures, ERFs and prior warming for each selection, in the same hierarchy without the levels that do not apply. `observations/` also holds `Obs_timeseries_*.csv`: the reference temperatures each run used (the selected members, with the pre-industrial offset removed), which `combine_results_iterations.py` plots.

The CSVs are:
- **Timeseries**: Annual values for all variables and percentiles
- **Headlines**: Values for specific headline definitions (single year, SR1.5 30-year average centred using extrapolations, AR6 lagged decade average, CGWL 20 year average centred using constrained projections)
- **Rates**: Warming rates for all variables and percentiles (if calculated)

Column structure: Multi-index with (variable, percentile) pairs, including various percentiles (5th, 17th, 50th, 83rd, 95th, etc.) representing the uncertainty distribution.

The figures in `plots/aggregated/` follow the same `ERF--…_GMT--…`/`ENSEMBLE-MEMBER--`/`VARIABLES--` hierarchy, and their names carry the same labels. For each regressed range (`REGRESSED-YEARS--{start}-{end}/`):
- `Timeseries_*` = the attributed warming timeseries, with uncertainty, against the observations
- `Rates_*` = warming rates (if `INCLUDE_RATE=y`)
- `SPM2_BarPlot_*` = one per headline period: a bar chart of the attributed warming, in the style of IPCC AR6 WGI Figure SPM.2
- `Waterfall_BarPlot_*` = one per headline period: the contributions to total warming as a waterfall, with subtotals

And for each selection as a whole:
- `Timeseries-animation_*.gif` = the timeseries figures as an animation across the regressed ranges
- `Historical_only_<headline>_*`, `Historical_only_headlines_*`, `Historical_and_full_headlines_*`, `ANNUAL_Historical_vs_Full_timeseries_*` = the historical-only timeseries for each headline definition, compared with each other and with the full-information GWI
- `Historical_delta_contributions_*` = the sources of the change in the GWI from each year to the next
- `Projected_warming_in_<year>_*` = the projected warming constrained by the final regressed range

Alongside these, `plots/iterations/` holds `Compare-Iterations_*` figures showing the spread between repeat iterations (only where there are two or more), and `plots/erfs/` and `plots/priors/` the ERF and prior timeseries.

##### 2.4.5. Data Inputs

Input data are in `data/`, each in a folder `data/<collection>/<kind>/<name>/`. Observations, and the ERFs that go with them, are filed by the year their history ends: `data/observed-<YYYY>/ERF/<product>/` (`IGCC`, `SSP_extension`) and `data/observed-<YYYY>/GMT/<product>/` (`HadCRUT`, `JK_multi`). Climate model test cases are filed by experiment, then model: `data/SMILE_ESM/{ERF,GMT}/MPI-ESM1.2-LR/` and `data/FutureVolc_ESM/{ERF,GMT}/NorESM1-M/`. The CMIP6 piControl simulations are in `data/piControl/GMT/CMIP6/`. Each data folder has a `SOURCE.md` giving the dataset, how to cite it, its licence and where it came from.

Each climate model's source file is in its folder in `models/` (see [Climate Models](#246-climate-models)).

```markdown
**Dataset:** <name and version>

**Cite:** <reference, with DOI>

**Licence:** <licence>

**From:** <URL or DOI, or who supplied it; date>
```

To add a new year of an existing dataset (e.g. `HadCRUT-2026`), put its files in `data/observed-2026/GMT/HadCRUT/`, named as for earlier years (the version number in a HadCRUT file name may change), with a `SOURCE.md`: no code changes are needed. A new product needs its own loader in `src/input_datasets.py`; the comment at the top of that file describes the dataset IDs and folders.


##### 2.4.6. Climate Models

The prior temperature responses come from a climate model, and the rest of the
attribution doesn't depend on which one. Each model has its own folder in
`models/`, named after its `model_choice` in `gwi.py`, holding the model's code
or parameters and a `SOURCE.md`:

- `models/FaIR_V2/`: FaIR v2.0.0-alpha1, used for all published results, with
  parameters emulating 28 CMIP6 models (`FaIR_CMIP6_climresp.csv`). The GWI
  runs every parameterisation, in parallel, so the spread between them is part
  of the climate parameter uncertainty. The FaIR code is downloaded from Zenodo
  into `models/FaIR_V2/FaIRv2_0_0_alpha1/` the first time `models` is imported,
  and is not tracked by git.
- `models/AR5_IR/`: the IPCC AR5 impulse response model, from earlier versions
  of the GWI. It is not wired into the current analysis.


##### 2.4.7. Memory and Computing Resources

The uncertainty calculation is heavily memory-constrained. `gwi.py` holds every attributed ensemble member in one array, whose size grows steeply with the number of samples per uncertainty source (`SUBSAMPLE_ITERATIONS`), and its peak memory use is about twice that array.

To help choose an ensemble size that fits, each `gwi.py` log reports, under `MEMORY USAGE ANTICIPATED`, the size of that array, the peak to expect, and the memory Slurm allocated to the job, and says whether it expects to run out. A run whose array cannot fit at all stops there, before any work is done. At the end of the run, `MEMORY USAGE RECORDED` gives the peak actually reached. Lower the number of samples, or raise `CPU_MEM`, until the anticipated peak fits.

Both `gwi.py` and `combine_results_iterations.py` run as many worker processes as the CPUs Slurm allocates to the job, or, outside Slurm, as the CPUs of the machine; set the `GWI_NUM_WORKERS` environment variable to use a different number. `combine_results_iterations.py` needs far less memory than `gwi.py`, though projects requiring exceptionally large numbers of different scenarios, or individual member analyses, may require more resources than an interactive slurm node's default allocation: an interactive job with 8 CPUs and 2 GB per CPU has been enough for our largest sets.


##### 2.4.8. Running the GWI for Individual Ensemble Members

Normally (`ERF_MEMBERS=all`, `GMT_MEMBERS=all`) every member of each uncertainty source is sampled within the one Monte Carlo calculation. Some projects instead need the full GWI calculated separately for each member of a dataset: for example, the Thorne et al. statistical analysis needed a GWI for each realisation of the observed temperatures. To do this:
1. Set the members wanted on one side (e.g. `GMT_MEMBERS={1..100}`), and `all` on the other; or, to pair ERF and GMT members one-to-one (as `FutureVolc_ESM-rcp45-Volc` needs), the pairing keyword on the other (e.g. `ERF_MEMBERS=paired_with_GMT_member`).
2. Set `CALCULATE_PRIORS_OUTPUT=n` if only temperature members are selected, since the priors would be the same for every GMT member.
3. Allow enough `WALLTIME`: each job runs `gwi.py` once for every member, one after another.

Each member's results go to their own `ENSEMBLE-MEMBER--{selection}` folder (e.g. `ENSEMBLE-MEMBER--ERF-all_GMT-7`), and `combine_results_iterations.py` processes them all. As a full set of figures for every member would run to tens of thousands, it thins the figures of member-by-member selections by default (`--plot-every=10`): every member gets the figures for its final regressed range, and one member in ten also gets them for every tenth regression end year. The historical-only figures are drawn for every member.
