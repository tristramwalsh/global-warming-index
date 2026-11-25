# The Global Warming Index

[![DOI: 10.5281/zenodo.24020859](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24020859-blue.svg)](https://doi.org/10.5281/zenodo.24020859)
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

It is best to run the GWI calculations using `run-gwi.sh`, a wrapper that builds every attribution configuration required and either submits each as a Slurm job or runs them in the current shell. Each run from `run-gwi.sh` is delivered through `gwi.py`, which calculates the GWI once for the options given on its command line. While `gwi.py` can be run directly, it takes about twenty options, and most analyses require running it many times across different configurations, usually on a Slurm-managed HPC cluster, which is why the `run-gwi.sh` wrapper is provided for ease of use and reproducibility.

To configure and run an attribution analysis you don't need to edit `run-gwi.sh`; the script `run-gwi.sh` takes its settings from a *recipe*: a short file in `recipes/` that sets every setting you need for a set of runs. Keeping each set of runs in its own recipe makes it easy to rerun, test and reproduce them.


#### 2.1. Basic Usage

##### 2.1.1. Run the GWI iterations
Make a recipe by copying the template, which lists and explains every setting (see **Configuration Arguments** below), and edit the values:
```bash
cp recipes/TEMPLATE.sh recipes/my-recipe.sh
```
Then activate the `gwi` environment, and run the recipe:
```bash
bash run-gwi.sh recipes/my-recipe.sh [dry|direct|slurm]
```
Run `bash run-gwi.sh` with no arguments for a usage message, which also lists the recipes available. The script checks that the `gwi` environment is active before running anything; it does not activate it for you, as that differs between machines.

A recipe sets every setting in the template, each once, and nothing else; the script checks this before running anything, and stops if a setting is missing, repeated or unknown (a typo, say). Comments are optional, so a recipe can be just the list of settings. When a new setting is added to the template, add it to every recipe in `recipes/` too. `recipes/dev*.sh` are ignored by git: use them for your own development runs.

Several recipes can share the same `OUTPUT_TAG`. This is useful when one analysis needs runs that a single recipe can't describe: each recipe sets one ERF dataset, one temperature dataset, one set of regressed variables and so on, so an analysis that compares several of them needs one recipe for each. Give all of those recipes the same tag and run each of them. Their results all go into the same `results_<tag>/` folder, and the combination script then processes them together as one analysis (see **2.1.2**).

In direct and slurm mode, a copy of the recipe goes into the log folder, headed with the date, the host, the mode, and the git commit the run started from (noting any uncommitted changes to tracked files).

**Run modes** (the second argument):
- `slurm` = generate and submit one SLURM job per configuration. Use this for full-size runs.
- `direct` = run each configuration now, one after another, in this shell: on a laptop or on an interactive node. Use this for quick tests and development. It stops at the first failure, and logs go to `logs_direct/` (or `logs_direct_<tag>/`).
- `dry` (the default) = print the `gwi.py` commands that would run, and run nothing.

For a quick test, run the template itself, whose values are a small, fast configuration (on a laptop this takes well under a minute):
```bash
bash run-gwi.sh recipes/TEMPLATE.sh direct
```

> [!WARNING]
> The template's results are not fit for any real use. Its ensemble is trivially small (10,000 members), so it only checks that everything runs. Real results need much larger ensembles (see `SAMPLING_ITERATIONS`).

##### 2.1.2. Combine results from multiple iterations
Once the runs have completed, run the combination script with the recipe's `OUTPUT_TAG` (`run-gwi.sh` prints this command when it finishes):

```bash
python combine_results_iterations.py --output-tag=quick-test
```

Alternatively, `--recipe=recipes/TEMPLATE.sh` reads the tag from the recipe for you.

This averages the repeat iterations, where applicable derives the historical-information-only timeseries (when a sequence of regression end years are used), and produces all associated figures.

See **Combining Results from Multiple Iterations** for its options, and **Output Structure** for where everything is written.

#### 2.2. Configuration Arguments

The following settings control the analysis. Every recipe sets each of them, named with a `RECIPE_` prefix (e.g. `RECIPE_ERF_DATASET=IGCC-2025`), so that it is clear in `run-gwi.sh` which values come from the recipe; the prefix is left out below. `recipes/TEMPLATE.sh` lists them all, with the values each can take.

**`START_REGRESS`**: The starting year for the observational constraint period (e.g., `1850`)
- Defines when start year of the data ranges that the regression constraint is applied over.


**`END_REGRESS`**: The ending year(s) for observational constraint period
- Full-information GWI: For a single end year: just specify one value
    - e.g. `2023` creates a single job with regression from 1850-2023
- Historical-only GWI: To generate GWI so that the GWI in year Y is based only on information up to and including year Y, you need to repeatedly re-run the GWI analysis with the regression constraint ending in every year of the historical-only timeseries.
    - e.g. `seq 1950 2024` creates separate jobs for each year from 1950-2024, each only incorporating years from START_REGRESS to END_REGRESS in the constraining.


**`SAMPLING_ITERATIONS`**: How to sample the ensemble, one entry per iteration
- Assuming independence, the complete ensemble is every combination of the members of the four uncertainty sources: about 790 million members for the IGCC assessment (28 FaIR parameter sets × 1000 ERF members × 200 HadCRUT members × 141 piControl segments), all equally weighted. A run rebuilds a sample of $N$ of them, chosen by a *sampling design*, or every one (see **What the Scripts Do**).
- Each entry is a design and the ensemble size $N$ joined by a dash, e.g. `latin-1e6`, or `all`. $N$ is a whole number, or in e notation (`1e6` = 1,000,000). The designs, each treating the four sources alike:
    - `latin` = $N$ members in which every member of each source is used equally often (to within one), the sources paired at random (a Latin hypercube), and no member drawn twice until every member has been drawn. The balance removes most of the sampling variance that Monte Carlo has, and drawing without repeats lowers the rest by the share of the complete ensemble drawn: so at the complete ensemble's size the sample *is* the complete ensemble, the same as `all`, and beyond it every member is drawn equally often (see `tests/plot_latin_sampling.py` for a picture).
    - `mc` = $N$ members drawn independently and uniformly from the complete ensemble (Monte Carlo).
    - `cube` = every combination of $k$ random members of each source (all of a source with fewer than $k$), with the largest $k$ that keeps the cube within $N$, so it can have fewer members than asked for (by up to ~20%). Earlier GWI releases used a cube that kept every parameter set, with $k$ given directly.
    - `all` = every member of the complete ensemble: no sampling error, and the same result every time. It needs a large-memory node for IGCC-sized datasets (see **Memory and Computing Resources**).
- The number of entries is the number of iterations. When repeat runs are generated, the results are combined using a weighted average (weighted by ensemble size of each iteration) in `combine_results_iterations.py`, to minimise sampling noise. If one configuration's iterations mix designs or ensemble sizes (usually two pieces of work sharing an `OUTPUT_TAG`, or none set), the combination script still averages them, but prints a warning listing them, e.g. `mixes ensemble designs/sizes: latin-1000000 (3 iterations), mc-30000 (1 iteration)`.
- Examples:
    - e.g. `(latin-1e4)` = a quick test
    - e.g. `(latin-1e6 latin-1e6 latin-1e6)` runs 3 repeat iterations of a million members each
    - e.g. `(latin-1e5 latin-1e6 latin-1e7)` runs 3 iterations of increasing size
- The results files name the design, the actual ensemble size and the seed in one field (e.g. `ENSEMBLE--latin-1000000-seed42`), and each run's log says how many members of each source its sample uses.

**`SEED`**: A seed for the random sampling of ensemble members, for exactly reproducible runs
- Leave empty (the usual setting) for every run to draw its own seed, and so sample afresh: repeat iterations differ, and averaging them reduces the sampling noise. The seed drawn is logged, and named in the results files, so any run can be reproduced.
- Set to a whole number (e.g. `42`) to draw exactly the same samples each time: runs of the same configuration on the same inputs then give identical results. This is useful for checking that a code change leaves the results unchanged.
- With a seed set, each iteration in `SAMPLING_ITERATIONS` uses the seed plus its number less one (`SEED=42` gives seeds 42, 43, 44, ... to iterations 1, 2, 3, ...), so repeat iterations draw different members, and any one of them can be re-run on its own from the `gwi.py` command in its log. The end years of one iteration share its seed, so they draw the same members. Consecutive seeds give independent draws: numpy hashes each seed into its own stream of random numbers.


**`PREINDUSTRIAL_ERA`**: Pre-industrial baseline period
- Used to define the baseline pre-industrial proxy for temperatures to be produced relative to.
- Examples:
    - `1850-1900` (default) = 1850-1900 average as pre-industrial baseline
    - `1981-2010` = 1981-2010 average as pre-industrial baseline
    - `n` = no pre-industrial offset applied (temperatures are left in their default state, which may be relative to a different baseline depending on the dataset)

**`INCLUDE_REG_CONST`**: Toggle regression constant (`y` or `n`)
- Specifies whether to include a constant term offset within the multi-variable regression. We typically set this to `n`, as in `recipes/TEMPLATE.sh`. (`gwi.py` itself defaults to `y` if the option is not passed.)
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
- NOTE: Where CGWL headlines are required, the truncation range must extend far enough into the future to accommodate the 9 years after each year that the CGWL averages (see **Truncation vs Regression Ranges** below).


**`DEFINITIONS`**: Which warming definitions to give
- `'annual,SR1.5,AR6,CGWL,rates'` = all five: the warming in each year; the value at mid-year of the line fitted through the last 15 years (SR1.5); the mean of the last 10 years (AR6); the mean of the 19 years centred on the year (CGWL); and the warming rate, the slope of the line fitted through the last 10 years (AR6).
- Each definition gets its own timeseries CSV, over every output year whose window fits within the truncation range, and its values at the headline years go into the headlines CSV.
- At least one is needed, e.g. `annual` for the basic annual timeseries. The annual timeseries is always written, since the plots build on it.


**`HEADLINE_YEARS`**: Which years to calculate headlines for
- `'end_regress'` = use the end of regression range (recommended for historical-only)
- `'end_trunc'` = use the end of truncation range
- `'IGCC'` = the end of the regression range, plus each definition's year assessed in the IPCC reports, repeated to validate against them and to show how the assessment has changed since: 2017 for the annual and SR1.5 definitions (assessed in SR1.5), and 2010-2019 for the AR6 mean and rate (assessed in AR6)
- `'2024'` = specific single year
- `'2023,2024,2025'` = multiple specific years
- `'end_regress,2050,2100,2300'` to combine end_regress and manual years
- `$(seq -s, 1950 2024)` = all years in a range (expensive)


**`QUANTILE_STEP`**: The quantiles: every definition's timeseries as percentiles every this many percent, from the 0th to the 100th, in one netCDF file (`GWI_timeseries-quantiles`; see **Output Structure**)
- `0.1` = every 0.1 percentile, 1001 levels: the distribution finely, as the sampling studies compare it
- `1` = every percentile, 101 levels
- `none` (or empty) = no quantiles, only the summary CSVs (the median and likely and very likely ranges), which are always written
- Taken in the same pass as the summary percentiles, which they make about twice as slow; the file's size depends on the number of levels, not of members.


**`RELEASE_MEMBERS`**, **`RELEASE_METHOD`**, **`RELEASE_TARGET_VARIABLE`**, **`RELEASE_TARGET_YEAR`**: The release: a few of the run's members, chosen to be representative of its whole ensemble, written whole to one netCDF file (`GWI_timeseries-members`) for users to take as an equal-weight ensemble of their own. The run's ensemble itself (e.g. a million members) is never written member by member.
- `RELEASE_MEMBERS=(1000)` = one release of 1000 members (about 21 MB with every definition); `(100 1000 10000)` = one release of each size, from the same run; `(0)` = none. At most the run's number of members; a release over 500 MB stops the run before it starts.
- `RELEASE_METHOD=stratified` = one member from each of `RELEASE_MEMBERS` bands of equal probability of the release's *target*, one variable in one year (`RELEASE_TARGET_VARIABLE` in `RELEASE_TARGET_YEAR`, its annual value; the year stated, within the truncation range): that variable's distribution in that year is then as close to the run's as so few members can be (for 1000 members from a million, Ant's 5th percentile within ~0.0005 °C against ~0.006 °C for a random choice), and the other variables are as good as a random choice, or better if correlated with it. `random` = at random, a baseline to judge stratified against, not for use.
- The target goes into the file's name (e.g. `RELEASE--stratified-Ant2025-1000`) and attributes (`release_target`), so a release says what it was made for. The method is in `src/method/release.py`.


**`OUTPUT_TAG`**: A name for this piece of work (e.g. `IGCC2025`)
- Keeps separate pieces of work apart, so that they neither overwrite each other nor all get processed together: results go to `results_<tag>/`, figures to `plots_<tag>/` and logs (with a copy of the recipe) to `logs_slurm_<tag>/` (or `logs_direct_<tag>/` in direct mode).
- Usually the recipe's own name; recipes that make up one experiment share a tag.
- Pass the same tag to `combine_results_iterations.py` (`--output-tag=<tag>`); alternatively, give it one of the recipes (`--recipe=<recipe>`) and it reads the tag from there. `run-gwi.sh` prints the command when it finishes.
- Left empty, the plain `results/`, `plots/` and `logs_slurm/` (or `logs_direct/`) folders are used.


**Slurm resources** (slurm mode only): `CLUSTER`, `SIM_CPUS`, `CPU_MEM` and `WALLTIME`
- `CLUSTER` = which Slurm cluster to submit to on Oxford ARC: `arc`, `htc`, or `arc,htc` to take whichever starts sooner. The partition is set automatically from the host.
- `SIM_CPUS`, `CPU_MEM`, `WALLTIME` = CPUs, memory per CPU, and time limit for each job.
- Our standard runs use `SIM_CPUS=28`, `CPU_MEM=9000M` and `WALLTIME=0:10:00`, which suits ensembles of about 6 million members: such a run, with every definition, takes 3 to 5 minutes and peaks near 5 GiB. Both grow in proportion to the number of members (see **Memory and Computing Resources**); member-by-member runs need more time.

#### 2.3. What the Scripts Do

##### 2.3.1. `run-gwi.sh`

The `run-gwi.sh` script:
1. Checks the recipe against `recipes/TEMPLATE.sh`, then reads it
2. Loops over `END_REGRESS` values (regression end years) and `SAMPLING_ITERATIONS` (the sampling of each iteration)
3. For each combination, calls `gwi.py` with the specified CLI arguments (once per selected member, one after another, when `ERF_MEMBERS` or `GMT_MEMBERS` list members; the list of runs is built in `src/infrastructure/prepare_ensemble_selections.sh`):
    - in slurm mode, from a SLURM job named after the run's configuration, which it submits; each job's log goes to `logs_slurm/` (or `logs_slurm_<tag>/`)
    - in direct mode, immediately in the current shell; each combination's log goes to `logs_direct/` (or `logs_direct_<tag>/`)

##### 2.3.2. `gwi.py`

Each `gwi.py` run attributes the observed warming once, for one configuration, as a short pipeline of named steps (`main`, at the bottom of `gwi.py`; its module docstring lists them):

1. **Read the configuration and load the inputs**: the ERF ensemble ($E$ members), the observed temperatures ($O$ members), the piControl segments that sample internal variability ($C$), and the FaIR parameter sets ($P = 28$, one emulating each of 28 CMIP6 models).
2. **Draw the members.** A *member* of the ensemble is one choice from each source of uncertainty: a parameter set $p$, an ERF member $e$, an observed member $o$ and a piControl segment $c$. The complete ensemble is every combination, $P \cdot E \cdot O \cdot C$ members. The run chooses $N$ of them (`--ensemble-size`) with its sampling design (`--sampling-design`: `latin`, `mc`, `cube`, or `all` for every member; see `SAMPLING_ITERATIONS`), drawing from one random number generator seeded by `--seed` (drawn afresh, and logged, if not given). Each member is stored as its four positions (its *member index*), in 1 or 2 bytes each.
3. **Stage A, modelling warming priors**: FaIR runs on each forcing variable alone (e.g. GHG, OHF, Nat, and with `INCLUDE_SUB_VARS` their components), for every parameter set and ERF member, giving the modelled warming $T_\mathrm{mod}(p, e, t, v)$.
4. **Stage B, constraining warming**: for every (parameter set, ERF member) pair, the least-squares fit of the observed temperatures onto the modelled warming of the regressed variables, over the regression years, gives the scaling factors that constrain it. The fit is made once for each observed member ($\beta_\mathrm{obs}$) and once for each piControl segment ($\beta_\mathrm{pic}$, the spread that internal variability adds); since the fit is linear in its target, a member's scaling factors are the sum of the two, $\beta = \beta_\mathrm{obs}[o] + \beta_\mathrm{pic}[c]$.
5. **Stage C, drawing ensemble members**: member $n$'s attributed warming from each forcing variable is its scaling factor times its modelled warming, $\beta_n \, T_\mathrm{mod}(p_n, e_n, t, v)$ (a sub-variable takes the factor of the variable it is part of); `Tot` is their sum over the regressed variables (with the constant, if `INCLUDE_REG_CONST=y`), `Res` the observed temperature minus `Tot`, and `Ant` = GHG + OHF (see **Variable Regression Logic**). The members are never all held at once: `src/method/members.py` rebuilds them a block of years at a time, and only where each output needs them.
6. **The percentiles**: each warming definition (see `DEFINITIONS`) is applied to every member, and the percentiles taken across the members, in every year whose window fits: the summary's (the median and likely and very likely ranges), and with them, if `QUANTILE_STEP` is set, the quantiles'. The definitions are described once, in `WARMING_DEFINITIONS` in `gwi.py`, and applied in the same way to the observations. The prior warming (stage A's, before the constraint) and the ERFs have percentile summaries output as well.
7. **The release**, if `RELEASE_MEMBERS` asks for one: the members chosen (see `RELEASE_METHOD`), and just those rebuilt over every year of every definition.
8. **Write** the inputs as used (the observed members and the piControl segments), the priors and the ERFs, then each definition's timeseries and the headlines, the quantiles and the releases (see **Output Structure**).

The run's log is divided into labelled sections (`RUN CONFIGURATION`, `ENSEMBLE SAMPLING`, `STAGES A, B AND C`, `MEMORY USAGE ANTICIPATED`, `DEFINITIONS`, `DEFINITIONS FOR OBSERVATIONS`, `PRIORS AND ERF`, `MEMORY USAGE RECORDED` and `TOTAL`), with the time each step took.


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

**Important for the warming definitions**: 
- Each definition needs its window of years to lie within the truncation range:
  - **SR1.5**: the target year and the 14 years before it (the end point of a 15-year linear trend, equivalent to a 29-year mean centred on the year)
  - **AR6**: the target year and the 9 years before it (10-year lagged mean)
  - **CGWL**: the target year and the 9 years either side of it (19-year centred mean)
  - **rates**: the target year and the 9 years before it (the slope of the line fitted through those 10 years)
- Each definition's timeseries covers the years whose window fits. A headline year whose window doesn't fit is left out of the headlines, with a note in the log.
- Note that the definitions for SR1.5 and CWGL technically prescibe a 30 year and 20 year mean respectively. We choose to use 29 and 19 years respectively, so that the centred mean lands on is the mid-point of a year, rather than the end-point for consistency with with the annual mean and the AR6 decade mean, which are both indexed as the mid-point of a calendar year.
  
If you set `HEADLINE_YEARS='end_regress'`, make sure your truncation range extends far enough to accommodate these definitions. For example, if `end_regress=2024` and you want CGWL, you need `end_trunc >= 2033`, and forcing projections that extend that far.

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

All options are optional:
- `--output-tag=<tag>` = the recipes' shared `OUTPUT_TAG`, so that the script reads and writes that piece of work's folders. Default: none (the plain `results/` and `plots/`).
- `--recipe=<recipe>` = an alternative to `--output-tag`: a recipe file, e.g. `recipes/TEMPLATE.sh`, whose `OUTPUT_TAG` the script reads. Recipes in one experiment share a tag, so any of them will do. Give this or `--output-tag`, not both.
- `--re-calculate=y|n` = `y` (default) recomputes the averages of the iterations, and the historical-only timeseries derived from them, and saves both to the aggregated folder. `n` reads both back from the aggregated folder, as written by an earlier `y` run, and only draws the figures.
- `--plot-every=N` = thins the figures of member-by-member selections (see **Running the GWI for Individual Ensemble Members**). Default `10`; `1` draws every figure. The standard `ERF-all_GMT-all` selection is always plotted in full.

###### 2.4.3.1 Aggregating Iteration Outputs
For iterations generated with `SAMPLING_ITERATIONS`, the script:
1. Finds all iterations generated from `SAMPLING_ITERATIONS` in `results/iterations/`
2. Combines them using a weighted average (weighted by ensemble size)
3. Saves combined results to `results/aggregated/`

(With a tag set, these are `results_<tag>/iterations/` and `results_<tag>/aggregated/`.)

###### 2.4.3.2 Historical-Only Timeseries
For iterations generated in order to provide different regression end years (e.g., `END_REGRESS=$(seq 1950 2024)`):
1. For each regression end year (e.g., 1950, 1951, ..., 2024), a separate GWI calculation is performed
2. `calculate_historical_only()` (in `src/postprocessing/historical_only.py`) extracts values for each year from the appropriate regression period
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
│                        ├── GWI_timeseries-summary-ANNUAL_*.csv
│                        ├── GWI_timeseries-summary-<DEFINITION>_*.csv  (SR15, AR6, CGWL, RATES: each definition asked for)
│                        ├── GWI_headlines-summary_*.csv
│                        ├── GWI_timeseries-quantiles_*.nc  (if QUANTILE_STEP is set)
│                        └── GWI_timeseries-members_*_RELEASE--*.nc  (one per size in RELEASE_MEMBERS)
├── plots/        figures drawn by combine_results_iterations.py (see below)
├── logs_slurm/   one log per Slurm job, and a copy of each recipe run (slurm mode)
└── logs_direct/  one log per configuration, and a copy of each recipe run (direct mode)
```

**File names.** Each file's name starts with what it holds: `<source>_<years>-<form>[-<DEFINITION>]_`, then the configuration. The *source* is `GWI` for the results (the attributed warming), `PRIOR` for the interim prior warming (FaIR's, before the constraint), or `ERF`, `Obs` or `PiControl` for the inputs as the run used them (the forcing, the observed temperatures, the piControl segments), which are processed only to be plotted beside the results, by the combination script, on the website and in papers. The *years* are the whole `timeseries` or the `headlines` years; the *form* is how the distribution across the members is given. The results:

| | **summary** (CSV): median, likely and very likely ranges | **quantiles** (netCDF): percentiles every `QUANTILE_STEP` | **members** (netCDF): representative members |
| --- | --- | --- | --- |
| **timeseries** (every year) | `GWI_timeseries-summary-<DEFINITION>` (one per definition) | `GWI_timeseries-quantiles` (every definition) | `GWI_timeseries-members` (the release, one file per size) |
| **headlines** (the headline years) | `GWI_headlines-summary` (every definition) | – (select the years from the timeseries) | – (select the years from the timeseries) |

The interim results and the inputs follow the same scheme: `PRIOR_timeseries-summary` (the prior warming), `ERF_timeseries-summary` (the forcing), `Obs_timeseries-members` (the observed members used), `Obs_timeseries-summary-<DEFINITION>` and `Obs_headlines-summary` (each definition applied to the observed members), and `PiControl_timeseries-members` (the piControl segments used, after filtering). The configuration that follows is the inputs, the regressed variables, the sample (`ENSEMBLE--<design>-<N>-seed<seed>`, which tells iterations apart: a run repeated with the same seed gives the same results, and replaces its files) and the regressed years, as each file needs.

The top level names the run's inputs: the ERF and GMT dataset IDs and the pre-industrial period (`PREINDUSTRIAL--none` with no offset), plus `_COMMITTED--{start}-{end}` when `COMMITTED` is set; e.g. `ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900`. Below it, `ENSEMBLE-MEMBER--` gives the member selection, ERF first, e.g. `ERF-all_GMT-all` or `ERF-all_GMT-7`. Each file's name repeats these labels.

`{category}` is one of:
- `iterations/` = the results of each `gwi.py` run
- `aggregated/` = those averaged across iterations by `combine_results_iterations.py`, plus the historical-only timeseries
- `observations/`, `erfs/`, `priors/` = the observed temperatures, ERFs and prior warming for each selection, in the same hierarchy without the levels that do not apply (the ERFs and priors keep `VARIABLES--`, since their variables are the forcing variables the regressed ones give). The prior warming is FaIR's warming before the observational constraint: it depends only on the ERF ensemble and the FaIR parameters, not on the reference temperatures, and is summarised over every (FaIR parameter set, ERF member) pair, each counted once. Depending on the size of the GWI ensemble, those pairs can differ from the ones the GWI's members use; each run's log says how many it used. The priors, ERF, observed-timeseries and piControl files are the same for every iteration of a configuration, so they have no sample in their name, and each run overwrites them in one step, so that runs writing them at the same time never leave a partial file. `observations/` also holds `Obs_timeseries-members_*.csv`: the reference temperatures each run used (the selected members, with the pre-industrial offset removed), which `combine_results_iterations.py` plots; and `Obs_timeseries-summary-<DEFINITION>_*.csv`: each warming definition the run asks for, applied to every observed member, as percentiles across the members, in every year whose window lies within the observed years (the observed headlines, per regressed range, are the rows of all five definitions at the headline years, whatever the run asks for, so that runs sharing the observations write the same file). On observations the definitions keep the names they have for the attributed warming, although SR1.5's and the CGWL's meanings are strictly for attributed (human-induced, or projected) warming.
- `picontrol/` = the piControl segments each set of inputs sampled internal variability from, after filtering (`PiControl_timeseries-members_*.csv`, one per inputs folder), for the internal-variability figures.

**Output formats.** Summaries are CSV; distributions are netCDF (`src/outputs/distributions.py`):
- A *summary* is the median and the likely and very likely ranges (the 50th, 17th–83rd and 5th–95th percentiles across the members) of every variable in every year: a small table, readable in a spreadsheet, whose configuration is in its folder and file names.
- A *distribution* is the ensemble's distribution in full: its *quantiles* (the percentiles at many levels, e.g. every 0.1 from the 0th to the 100th), or members themselves. It is netCDF (float32, compressed), with named dimensions, and it carries its configuration as attributes, since a distribution file is often sent on by itself: the input datasets and their members, the pre-industrial era, the regressed variables and years, the output years, the sampling design, the ensemble size $N$ and the seed, the git commit and the `gwi.py` command, with a `long_name` and `units` for every variable (`K` for warming, a temperature difference, following the [CF conventions](https://cfconventions.org/), 1.11). Every netCDF file is written by one function, which refuses a file missing any of these.
- The one exception: the inputs as the run used them (the observed members and the piControl segments), which are written only to be plotted beside the results, are member tables in CSV, as the inputs come.

The CSVs are:
- **Timeseries** (`timeseries-summary-ANNUAL`, `-SR15`, `-AR6`, `-CGWL`, `-RATES`): each warming definition's values for all variables and percentiles, in every year whose window fits (the annual values; the SR1.5 trend end point, equivalent to a 29-year mean centred using extrapolation; the AR6 lagged decade mean; the CGWL 19-year mean centred using constrained projections; the AR6 rate, in °C per year)
- **Headlines**: every definition's values at the headline years, in one table

The netCDF files are:
- **Quantiles** (`timeseries-quantiles`, when `QUANTILE_STEP` is set): every definition's timeseries as percentiles every `QUANTILE_STEP` percent from the 0th to the 100th (1001 levels for 0.1), for every variable: one data variable per definition (`ANNUAL`, `SR15`, `AR6`, `CGWL`, `RATES`, as asked for), each with dimensions (`year`, `percentile`, `output_variable`), on the years of them all (NaN where a definition's window doesn't fit). The summary CSVs' percentiles are among them, identical to the last bit. The 0th and 100th percentiles are the smallest and largest members, which spread further with more members; the levels within them settle as the ensemble grows. For example, in Python: `xr.open_dataset(path)['AR6'].sel(year=2025, output_variable='Ant')` is the AR6 Ant distribution in 2016-2025.
- **Members** (`timeseries-members`, the release, one file per size in `RELEASE_MEMBERS`, named with its `RELEASE--<method>[-<variable><year>]-<size>` field): the released members' values, every year of every definition given, for every variable: one data variable per definition, as in the quantiles file, each with dimensions (`year`, `member`, `output_variable`). Along `member`, each member's FaIR parameter set, ERF member, observed member and piControl segment (`param_member`, `erf_member`, `gmt_member`, `pic_member`), so every member can be traced. The attributes add the release's size, method, what it was stratified on and how to use it: every member has equal weight; use them all, or the first of them, any number (the members are in an order in which every first k is spread across the distribution, see `src/method/release.py`); don't weight or resample them; a member is one consistent draw across all its years, variables and definitions.

Column structure: Multi-index with (variable, percentile) pairs, including various percentiles (5th, 17th, 50th, 83rd, 95th, etc.) representing the uncertainty distribution.

The figures in `plots/aggregated/` follow the same `ERF--…_GMT--…`/`ENSEMBLE-MEMBER--`/`VARIABLES--` hierarchy, and their names carry the same labels. For each regressed range (`REGRESSED-YEARS--{start}-{end}/`):
- `Timeseries_*` = the attributed warming timeseries, with uncertainty, against the observations
- `Rates_*` = warming rates (if `rates` is among the `DEFINITIONS`)
- `SPM2_BarPlot_*` = one per headline period: a bar chart of the attributed warming, in the style of IPCC AR6 WGI Figure SPM.2
- `Waterfall_BarPlot_*` = one per headline period: the contributions to total warming as a waterfall, with subtotals

And for each selection as a whole:
- `Timeseries-animation_*.gif` = the timeseries figures as an animation across the regressed ranges
- `Historical_only_<headline>_*`, `Historical_only_headlines_*`, `Historical_and_full_headlines_*`, `ANNUAL_Historical_vs_Full_timeseries_*` = the historical-only timeseries for each headline definition, compared with each other and with the full-information GWI
- `Historical_delta_contributions_*` = the sources of the change in the GWI from each year to the next
- `Projected_warming_in_<year>_*` = the projected warming constrained by the final regressed range

Alongside these, `plots/iterations/` holds `Compare-Iterations_*` figures showing the spread between repeat iterations (only where there are two or more), `plots/erfs/` and `plots/priors/` the ERF and prior timeseries, and `plots/picontrol/` two diagnostic figures of the piControl segments used for internal variability, against a rough estimate of the observed internal variability: `piControl_Selected_Segments_*` (running means) and `piControl_Internal_Variability_Distribution_*`.

##### 2.4.5. Data Inputs

Input data are in `data/`, each in a folder `data/<collection>/<kind>/<name>/`. Observations, and the ERFs that go with them, are filed by the year their history ends: `data/observed-<YYYY>/ERF/<product>/` (`IGCC`, `SSP_extension`) and `data/observed-<YYYY>/GMT/<product>/` (`HadCRUT`, `JK_multi`). Climate model test cases are filed by experiment, then model: `data/SMILE_ESM/{ERF,GMT}/MPI-ESM1.2-LR/` and `data/FutureVolc_ESM/{ERF,GMT}/NorESM1-M/`. The CMIP6 piControl simulations are in `data/piControl/GMT/CMIP6/`. Each data folder has a `SOURCE.md` giving the dataset, how to cite it, its licence and where it came from.

Each climate model's source file is in its folder in `models/` (see [Climate Models](#246-climate-models)).

```markdown
**Dataset:** <name and version>

**Cite:** <reference, with DOI>

**Licence:** <licence>

**From:** <URL or DOI, or who supplied it; date>
```

To add a new year of an existing dataset (e.g. `HadCRUT-2026`), put its files in `data/observed-2026/GMT/HadCRUT/`, named as for earlier years (the version number in a HadCRUT file name may change), with a `SOURCE.md`: no code changes are needed. A new product needs its own loader in `src/inputs/input_datasets.py`; the comment at the top of that file describes the dataset IDs and folders.


##### 2.4.6. Climate Models

The prior temperature responses come from a climate model, and the rest of the
attribution doesn't depend on which one. Each model has its own folder in
`models/`, holding the model's code or parameters and a `SOURCE.md`; the GWI
reaches them only through `src/method/emulators.py`:

- `models/FaIR_V2/`: FaIR v2.0.0-alpha1, used for all published results, with
  parameters emulating 28 CMIP6 models (`FaIR_CMIP6_climresp.csv`). The GWI
  runs every parameterisation, in parallel, so the spread between them is part
  of the climate parameter uncertainty. The FaIR code is downloaded from Zenodo
  into `models/FaIR_V2/FaIRv2_0_0_alpha1/` the first time `models` is imported,
  and is not tracked by git.
- `models/AR5_IR/`: the IPCC AR5 impulse response model, from earlier versions
  of the GWI, with its default parameters (`AR5_IR_params.py`). It is not
  wired into the current analysis.


##### 2.4.7. Memory and Computing Resources
The ensemble sizes for the GWI uncertainty calculations are extremely large, which can present significant memory and computing requirements. The code is written in a way to mitigate that pressure as far as possible.

<!-- `gwi.py` never holds the whole ensemble of attributed warming (see **What the Scripts Do**). What it holds throughout is small: the modelled warming of every (parameter set, ERF member) pair, the scaling factors of every observed member and piControl segment for each pair, the observed temperatures, and each member's four positions (1 or 2 bytes each). The members' attributed warming is rebuilt a block of years at a time, and a block, with the copy that the percentiles make of it, takes 2 × years per block × variables × members × 4 bytes. So the memory grows with the number of members, through the blocks, whose size is chosen to fit. -->

<!-- The years per block are as many as the threads can take percentiles of in one pass (⌊threads / variables⌋, e.g. 4 for 6 variables on 28 threads; bigger blocks are no faster), unless the memory free when the sample is made (stage C) allows fewer; the log's `Years per block` line says which limits it. For example, at about 6 million members (6 variables, 28 threads) the run holds 0.5 GiB, each block takes 1.1 GiB, and the measured peak is near 5 GiB, the inputs and stage A's FaIR runs coming on top. The complete ensemble (790 million members) needs about 19 GB for each year of a block (twice that with the percentiles' copy), and 4 GB for the members' positions. -->

Roughly, on 28 CPUs with every definition and the quantiles at 0.1 (`QUANTILE_STEP`): `latin-1e6` takes 1.5 minutes, `latin-1e7` 6 minutes and `latin-1e8` about 50 minutes, the time growing in proportion to the number of members. **`all` needs a large-memory node**: for IGCC-2025's 790 million members, about 160 GB, so that each block holds 4 years and every thread takes percentiles, and about 6 hours (with less memory the blocks shrink to 1 year, only one thread per variable takes percentiles, and the run is several times slower). The quantiles make large runs up to 1.6 times as long as without them.

To help choose an ensemble size that fits, each `gwi.py` log reports, under `MEMORY USAGE ANTICIPATED`, what is held throughout, the largest block, the peak to expect, and the memory Slurm allocated to the job. A run that cannot fit stops there, before the members are rebuilt. At the end of the run, `MEMORY USAGE RECORDED` gives the peak actually reached. Lower the ensemble size, or raise `CPU_MEM`, until the anticipated peak fits.

`gwi.py` rebuilds the members and takes their percentiles on as many threads, and `combine_results_iterations.py` runs as many worker processes, as the CPUs Slurm allocates to the job, or, outside Slurm, as the CPUs of the machine; set the `GWI_NUM_WORKERS` environment variable to use a different number. `combine_results_iterations.py` needs far less memory than `gwi.py`, though projects requiring exceptionally large numbers of different scenarios, or individual member analyses, may require more resources than an interactive slurm node's default allocation. <!-- An interactive job with 8 CPUs and 2 GB per CPU has been enough for our largest sets for the post-processing combination script. -->


##### 2.4.8. Running the GWI for Individual Ensemble Members

Normally (`ERF_MEMBERS=all`, `GMT_MEMBERS=all`) every member of each uncertainty source is sampled within the one calculation. Some projects instead need the full GWI calculated separately for each member of a dataset: for example, the Thorne et al. statistical analysis needed a GWI for each realisation of the observed temperatures. To do this:
1. Set the members wanted on one side (e.g. `GMT_MEMBERS={1..100}`), and `all` on the other; or, to pair ERF and GMT members one-to-one (as `FutureVolc_ESM-rcp45-Volc` needs), the pairing keyword on the other (e.g. `ERF_MEMBERS=paired_with_GMT_member`).
2. Allow enough `WALLTIME`: each job runs `gwi.py` once for every member, one after another.

Each member's results go to their own `ENSEMBLE-MEMBER--{selection}` folder (e.g. `ENSEMBLE-MEMBER--ERF-all_GMT-7`), and `combine_results_iterations.py` processes them all. As a full set of figures for every member would run to tens of thousands, it thins the figures of member-by-member selections by default (`--plot-every=10`): every member gets the figures for its final regressed range, and one member in ten also gets them for every tenth regression end year. The historical-only figures are drawn for every member.
