# The Global Warming Index

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) -->
<!-- [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/) -->

## Introduction

The Global Warming Index (GWI) is a method for attributing observed global temperature changes to different climate forcing components.

It uses a two-step regression approach: first, prior estimates of the temperature responses to individual forcings (greenhouse gases, other human forcing, and natural forcings) are generated using a climate emulator model (FaIR), then these modelled responses are constrained via regression against observed temperatures and pre-industrial control simulations to determine posterior estimate.

Very large Monte Carlo simulations (with $\mathcal{O}(10^8)$ ensemble members) are used to produce probabilistic estimates of human-caused warming, natural contributions, and total warming with full uncertainty quantification across multiple sources of uncertainty, including:
- radiative forcing uncertainty
- climate parameter uncertainty
- observed temperature uncertainty
- possible alternative realisations of internal variability

## Results Availability and Citations

The Global Warming Index is hosted by the Environmental Change Institute (ECI) at the University of Oxford, and is operationally updated as part of the Indicators of Global Climate Change (IGCC) project.

This repository does not contain pre-generated results, but provides the code and instructions to replicate the results.

The most recent results, published in the annual IGCC assessments can be found at:
- https://github.com/ClimateIndicator/anthropogenic-warming-assessment

If you use this code or results, these references should be cited:
- > Forster, P., Smith, C., Walsh, T. et al. (2025) *Indicators of Global Climate Change 2024: annual update of key indicators of the state of the climate system and human influence*. https://doi.org/10.5194/essd-17-2641-2025
    - The most recent published verison of the results from the GWI using this repository, with fully updated GWI methodology.
- > Haustein, K., Allen, M.R., Forster, P.M. et al. (2017) *A real-time Global Warming Index*. https://doi.org/10.1038/s41598-017-14828-5
    - The original reference for the original GWI method.


## Replicating the Results
Prerequisites:
- **Package Manager:** `Micromamba` (Conda may not solve dependencies for all systems)
- **HPC:** `SLURM` (The workflow is optimized for SLURM-based HPC clusters but can be adapted for local execution).

### 1. Setting up the environment
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/global-warming-index.git
   cd global-warming-index
   ```

2. Create a new environment with the required dependencies. We provide environment files for specific hosts (e.g., `arc-htc`, `ouce-linux`), but you can use them as a template:
   ```bash
   # Replace <host> with your system configuration file (e.g., environment_arc-htc.yml)
   micromamba env create -n gwi -f environment-<host>.yml
   micromamba activate gwi
   ```

### 2. Running the Analysis with SLURM Batch Processing

The main analysis is designed to run on HPC clusters using SLURM job submission. The workflow is controlled by the `schedule-gwi.sh` script, which generates and submits multiple SLURM job files to calculate the GWI for all required analysis configurations.

#### 2.1. Basic Usage

##### 2.1.1. Generate and submit GWI iteration jobs
```bash
bash schedule-gwi.sh
```
This script automatically generates and submits SLURM jobs. Before running, you should edit the key parameters at the top of `schedule-gwi.sh` (see **Configuration Arguments** below).

##### 2.1.2. Combine results from multiple iterations
If only one timeseries was generated, Step 1 is sufficient. However, in most cases, you will want to run multiple iterations to recombine later. Once the jobs have completed, run:

```bash
python combine_results_iterations.py --re-calculate=y
```

#### 2.2. Configuration Arguments

The following parameters in `schedule-gwi.sh` control the analysis:

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


**`PREINDUSTRIAL_ERA`**: Pre-industrial baseline period
- Used to define the baseline pre-industrial proxy for temperatures to be produced relative to.
- Examples:
    - `1850-1900` (default) = 1850-1900 average as pre-industrial baseline
    - `1981-2010` = 1981-2010 average as pre-industrial baseline
    - `n` = no pre-industrial offset applied (temperatures are left in their default state, which may be relative to a different baseline depending on the dataset)

**`INCLUDE_REG_CONST`**: Toggle regression constant (`y` or `n`)
- Specifies whether to include a constant term offset within the multi-variable regression. Typically left as `y`.
- Examples:
    - `y` = include the constant term in the regression (default)
    - `n` = do not include the constant term in the regression

**`VARS`**: Variables to regress on (comma-separated, no spaces)
- `GHG,OHF,Nat` = 3-way regression (greenhouse gases, other human forcings, natural)
- `Ant,Nat` = 2-way regression (anthropogenic total, natural)
- `Tot` = 1-way regression (total forcing only)

**`INCLUDE_SUB_VARS`**: Include sub-variables in output (`y` or `n`)
- Calculates component-wise contributions to the aggregate variables (e.g. GHG = CO2 + CH4 + N2O + F-gases).
- Useful for detailed attribution analysis while maintaining robust regression on aggregate variables.
- Note that these components must be included in the forcing dataset individually for this to be possible.


**`SCENARIO`**: Which dataset/scenario to analyse
- `observed-2024` = Latest observed temperatures and ERFs (to 2024). Available years:
    - 2023, 2024
- `observed-SSP<scen>` = Observed data (2024) with future SSP scenario extension. Available scenarios:
    - SSP119, SSP126, SSP245, SSP370, SSP585
- `observed_JK-2024-SSP<scen>` = John Kennedy's multi-dataset observation ensemble data, and historical ERFs to 2024 with `<scen>` extension so that CGWL headlines can be generated.
- `SMILE_ESM-SSP<scen>` = Single model initial-condition large ensemble. Available scenarios:
    - SSP126, SSP245, SSP370
- `NorESM_rcp45-Volc` = NorESM2-LM large ensemble with RCP4.5 forcing and volcanic eruptions included
- `NorESM_rcp45-VolcConst` = NorESM2-LM large ensemble with RCP4.5 forcing and volcanic eruptions excluded


**`COMMITTED`**: Calculate committed warming (`n` or `start-end`)
- Calculates the warming committed to occur by a future year if effective radiative forcing (ERF) is held constant from a specified start year.
- Format: `start_year-end_year`
- You can use `end_regress` as a keyword for the start year.
- Examples:
    - `n`: No committed warming calculation (default)
    - `2024-2300`: Hold ERF constant from 2024 to 2300
    - `end_regress-2300`: Hold ERF constant from the end of the regression period to 2300
- The scenario name in the output is appended with `_const-ERF_<start>-<end>`.
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


**`SPECIFY_ENSEMBLE_MEMBERS`**: Select specific ensemble members for Temperature and ERF.
- Examples:
    - `all` = use all available ensemble members (default)
    - `1` = use specific ensemble member number 1 (for example)
    - `{1..60}` = use specific member range from 1 to 60 (for example) (for specialized analyses)


**`SPECIFY_ENSEMBLE_MEMBER_SOURCE_FOR`**: Specify which dataset the ensemble member selection (above) applies to.
- `ERF` = Apply above selection to ERF only
- `GMT` = Apply above selection to GMT only
- `ERF,GMT` = apply above selection to both ERF and GMT; in this case, the same ensemble members will be paired for both sources; i.e. ensemble member i for ERF is paired with ensemble member i for GMT

#### 2.3. What the Script Does

The `schedule-gwi.sh` script:
1. Loops over `END_REGRESS` values (regression end years) and `SUBSAMPLE_ITERATIONS` (subsampling sizes)
2. Generates a SLURM job file for each combination
3. Each job calls `gwi.py` with the specified CLI arguments.
4. Submits the job to SLURM and cleans up the temporary job files


#### 2.4. Technical Notes

##### 2.4.1. Truncation vs Regression Ranges

The truncation range defines the output timeseries extent, while the regression range determines which years are used to fit the scaling coefficients that constrain the prior temperature response.

- **Regression range** (`--regress-range`): Years used to calculate the regression coefficients between modeled responses and observations
  - Must be within both the observed temperature data range AND the forcing data range
  - The code automatically clips this if you specify years outside available data
  
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
python combine_results_iterations.py --re-calculate=<y/n>
```

###### 2.4.3.1 Aggregating Iteration Outputs
For iterations generated with `SUBSAMPLE_ITERATIONS`, the script:
1. Finds all iterations generated from `SUBSAMPLE_ITERATIONS` in `results/iterations/`
2. Combines them using a weighted average (weighted by ensemble size)
3. Saves combined results to `results/aggregated/`

###### 2.4.3.2 Historical-Only Timeseries
For iterations generated in order to provide different regression end years (e.g., `END_REGRESS=$(seq 1950 2024)`):
1. For each regression end year (e.g., 1950, 1951, ..., 2024), a separate GWI calculation is performed
2. The `calculate_historical_only()` function extracts values for each year from the appropriate regression period
    1. When `headline` datasets are available, this pulls from the **`headlines`** dataframes at the appropriate string index
    2. When `headline` datasets are not , this pulls from the **`timeseries`** dataframes at the index `current_year`
3. The result is a timeseries where each year's attribution is based only on data available up to that year, which provides an estimate of what the GWI would have been if calculated in real-time each year, though note that the datasets and climate parameters are still the same as used in the full-information GWI.


### 2.4.4. Output Structure
Results are organized hierarchically:
```
global-warming-index/
└── results/
     └── {category}/
         └── SCENARIO--{scenario}/
             └── ENSEMBLE-MEMBER--{ensemble}/
                 └── VARIABLES--{vars}/
                     └── REGRESSED-YEARS--{start}-{end}/
                         ├── GWI_results_timeseries_*.csv
                         ├── GWI_results_headlines_*.csv
                         └── GWI_results_rates_*.csv  (if INCLUDE_RATE=y)
```

The CSVs are:
- **Timeseries**: Annual values for all variables and percentiles
- **Headlines**: Values for specific headline definitions (single year, SR1.5 30-year average centred using extrapolations, AR6 lagged decade average, CGWL 20 year average centred using constrained projections)
- **Rates**: Warming rates for all variables and percentiles (if calculated)

Column structure: Multi-index with (variable, percentile) pairs, including various percentiles (5th, 17th, 50th, 83rd, 95th, etc.) representing the uncertainty distribution.

## 2.4.5. Data Inputs

Input data is stored in the `data/` directory. This includes:
- **Temperatures:**
    - **Observations:** Temperature datasets (e.g., HadCRUT, CMIP simulation outputs).
    - **Simulated internal variability:** PiControl CMIP simulations
- **Forcings:** Effective Radiative Forcing (ERF) timeseries.

Ensure that the data corresponding to your `SCENARIO` selection in `schedule-gwi.sh` is present in the `data/` folder before running, and that an appropriate data loader function has been written in `definitions.py`.
