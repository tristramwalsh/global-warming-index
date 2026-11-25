"""Where results and figures are written, and what the files are called.

OUTPUT NAMES. Each file's name starts with what it holds:

    <source>_<years>-<form>[-<DEFINITION>]_<configuration>

    source       what it is of:
                   GWI        the attributed warming: the results
                   PRIOR      the prior warming, FaIR's before the
                              constraint: an interim result
                   ERF, Obs,  the inputs, as the run used them (the
                   PiControl  forcing, the observed temperatures, the
                              piControl segments), processed only so that
                              they can be plotted beside the results (by
                              the combine step, the website, papers); not
                              products for anyone else
    years        timeseries (every year) or headlines (the headline years)
    form         how the distribution across the members is given:
                   summary    the median and the likely and very likely
                              ranges (CSV)
                   quantiles  the percentiles at many levels, e.g. every
                              0.1 (netCDF)
                   members    the members themselves (netCDF; for the
                              inputs, CSV: see distributions.py)
    DEFINITION   the warming definition, for the summary timeseries, one
                 file each (ANNUAL, SR15, AR6, CGWL, RATES); the netCDF
                 files hold every definition

The results, the attributed warming:

                summary (CSV)        quantiles (netCDF)    members (netCDF)
    timeseries  GWI_timeseries-      GWI_timeseries-       GWI_timeseries-
                summary-<DEFINITION> quantiles             members (the
                                                           release: see
                                                           release_name)
    headlines   GWI_headlines-       -                     -
                summary

(The headline years' quantiles and members are in the timeseries files:
select the years.) Each release comes with GWI_headlines-fidelity, a CSV
comparing it with the run's whole ensemble at the headline years.

The interim results and the inputs, in the same scheme:

    PRIOR_timeseries-summary                    the prior warming
    ERF_timeseries-summary                      the forcing
    Obs_timeseries-members                      the observed members used
    Obs_timeseries-summary-<DEFINITION>,        each definition applied to
    Obs_headlines-summary                       the observed members
    PiControl_timeseries-members                the piControl segments used,
                                                after filtering

The configuration follows: the inputs, the regressed variables, the sample
(sample_name) and the regressed years, as each file needs. The combine
step's averages keep the names, ending in AVERAGE, and its historical-only
series are GWI_timeseries-summary-<DEFINITION>_HISTORICAL-ONLY_.
The names are kept short: a file name has at most 255 bytes, a limit of
the filesystem.
"""
import os
import re
import uuid

import xarray as xr


def tagged_folder(base, tag):
    """Return the output folder for a run, optionally tagged.

    Keeps separate pieces of work from overwriting one another, and from
    being swept up together by the combination script. Without a tag the
    folders are the plain ones they have always been, so existing behaviour
    and existing directories are untouched:

        tagged_folder('results', None)          -> 'results'
        tagged_folder('results', 'Thorne2025')  -> 'results_Thorne2025'
        tagged_folder('plots', 'IGCC')          -> 'plots_IGCC'

    Pass the same --output-tag to gwi.py and to combine_results_iterations.py
    so that the second reads what the first wrote.
    """
    if tag in (None, '', 'none'):
        return base
    if not re.fullmatch(r'[A-Za-z0-9._-]+', tag):
        raise ValueError(
            f'Invalid --output-tag {tag!r}. It becomes part of a directory '
            f'name, so use only letters, digits, dots, dashes and '
            f'underscores.')
    return f'{base}_{tag}'


# Where combine_results_iterations.py reads and writes. PLOT_FOLDER,
# AGGREGATED_FOLDER, ITERATIONS_FOLDER, PRIORS_FOLDER, ERFS_FOLDER,
# OBSERVATIONS_FOLDER and PICONTROL_FOLDER are created by set_output_folders()
# below -- the only place the layout is written down -- and read, as
# results_layout.<NAME>, by the combine script and its support modules as
# they run.


def set_output_folders(output_tag=None):
    """Point every output folder at this piece of work's own directory.

    With no tag these are the plain results/ and plots/ folders this script
    has always used. With one, they become results_<tag>/ and plots_<tag>/,
    so that separate pieces of work -- a paper, an assessment, a
    collaborator's runs, your own development -- neither overwrite each other
    nor all get processed every time this script runs.

    Pass the same tag that was given to gwi.py, so that this reads what that
    wrote.
    """
    global PLOT_FOLDER, AGGREGATED_FOLDER, ITERATIONS_FOLDER
    global PRIORS_FOLDER, ERFS_FOLDER, OBSERVATIONS_FOLDER, PICONTROL_FOLDER

    results_root = tagged_folder('results', output_tag)
    PLOT_FOLDER = tagged_folder('plots', output_tag) + '/'
    AGGREGATED_FOLDER = f'{results_root}/aggregated'
    ITERATIONS_FOLDER = f'{results_root}/iterations'
    PRIORS_FOLDER = f'{results_root}/priors'
    ERFS_FOLDER = f'{results_root}/erfs'
    OBSERVATIONS_FOLDER = f'{results_root}/observations'
    PICONTROL_FOLDER = f'{results_root}/picontrol'

    print(f'Reading results from {ITERATIONS_FOLDER}/, '
          f'writing plots to {PLOT_FOLDER}'
          f'{"" if output_tag else "  (no tag set)"}')


def get_subdirs(path, prefix):
    """Get sorted list of subdirectories starting with a prefix."""
    if not os.path.exists(path):
        return []
    return sorted([
        d.split(prefix)[1] for d in os.listdir(path)
        if d.startswith(prefix) and os.path.isdir(os.path.join(path, d))
    ])


def get_input_labels(path):
    """Sorted list of the results folders in path, each named after the
    inputs of the runs inside it,
    e.g. 'ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900'."""
    if not os.path.exists(path):
        return []
    return sorted([
        d for d in os.listdir(path)
        if d.startswith('ERF--') and os.path.isdir(os.path.join(path, d))
    ])


def gmt_id_from_label(scen):
    """The GMT dataset ID in an inputs label, e.g. 'HadCRUT-2025' from
    'ERF--IGCC-2025_GMT--HadCRUT-2025_PREINDUSTRIAL--1850-1900'."""
    return scen.split('_GMT--')[1].split('_PREINDUSTRIAL--')[0]


def sample_name(sampling_design, n_members, seed):
    """The field naming a run's sample in its results files, e.g.
    'ENSEMBLE--latin-1000000-seed42'.

    sampling_design:
        the design (see src/method/sampling.py)
    n_members:
        the sample's actual size, N, which iteration_averaging weights each
        iteration by (see ensemble_size_from_name)
    seed:
        the seed its members were drawn with

    One field, not three (SAMPLING--, ENSEMBLE-SIZE--, SEED--), because the
    results files' names are close to the 255-byte limit on a file name:
    with the longest dataset IDs (FutureVolc's), 1e8 members and a drawn
    ten-digit seed, three fields take a name to ~260 bytes, one to ~229.

    It also tells the iterations of a configuration apart, so the names
    need no date: a run repeated with the same seed draws the same members,
    and replaces its files with the same results.
    """
    return f'ENSEMBLE--{sampling_design}-{n_members}-seed{seed}'


def release_name(method, n_release, stratify_variable=None,
                 stratify_year=None):
    """The field naming a release in its file's name, e.g.
    'RELEASE--stratified-Ant2025-1000', or 'RELEASE--random-1000'.

    method:
        how its members were chosen (see src/method/release.py)
    n_release:
        M, its number of members
    stratify_variable, stratify_year:
        what a stratified release was stratified on, e.g. 'Ant' and 2025
        (in the name, so no one can miss what it was made for)
    """
    # What it was stratified on, if it was
    stratified_on = (f'-{stratify_variable}{stratify_year}'
                     if method == 'stratified' else '')
    return f'RELEASE--{method}{stratified_on}-{n_release}'


def sample_from_name(file_name):
    """The sample's design and size, N, from a results file's name (see
    sample_name).

    file_name:
        the name, or its path
    Returns (design, N):
        design: e.g. 'latin'; None for names written before the sampling
            designs (Phase 2 of the engine refactor), which give only the
            size, as ENSEMBLE-SIZE--<N>, and are read too, so older results
            can still be averaged
        N: an int
    """
    # The sample's field, ENSEMBLE--<design>-<N>-seed<seed>
    match = re.search(r'_ENSEMBLE--([a-z]+)-(\d+)-seed\d+', file_name)
    if match is not None:
        return match.group(1), int(match.group(2))
    # ... or the field of the names before it
    match = re.search(r'_ENSEMBLE-SIZE--(\d+)', file_name)
    if match is not None:
        return None, int(match.group(1))
    raise ValueError(f'No ensemble size in the file name {file_name!r}.')


def ensemble_size_from_name(file_name):
    """The sample's size, N, from a results file's name (see
    sample_from_name), which iteration_averaging weights each iteration by.

    file_name:
        the name, or its path
    """
    return sample_from_name(file_name)[1]


def write_atomically(path, write):
    """Write a file so that no one ever sees half of it.

    path:
        the file to write; replaced if it exists
    write:
        a function that writes the whole file to the path it is given, e.g.
        lambda temp_path: table.to_csv(temp_path)

    Several Slurm jobs can write the same file at once: run-gwi.sh submits
    one job per (iteration, regression end year), and every one of them
    writes the same undated files (the priors, the ERF percentiles, the
    observations and the piControl segments), each with identical content.
    Written in place, a file that another job is still writing, or that a
    job was killed while writing, can be read truncated.

    So the file is written in full under a temporary name in the same
    folder, then renamed onto `path`. On one filesystem a rename is atomic:
    a reader finds either the old complete file or the new complete file,
    never a partial one, and of several jobs writing the same file, the last
    to finish wins. The temporary name is short and random (output names
    are already close to the 255-byte limit) and hidden, ending in .tmp, so
    nothing that reads results picks it up.
    """
    # The folder the file goes in
    folder = os.path.dirname(os.path.abspath(path))
    # A temporary file beside it, under a name no other job will use
    temp_path = os.path.join(folder, f'.gwi-write-{uuid.uuid4().hex}.tmp')
    try:
        # Write the whole file under the temporary name
        write(temp_path)
        # Move it onto the final name, in one atomic step
        os.replace(temp_path, path)
    finally:
        # Remove the temporary file if anything stopped the move
        if os.path.exists(temp_path):
            os.remove(temp_path)


def write_csv(table, path, year_heading=None):
    """Write a summary to a CSV file, in one step (see write_atomically).

    table:
        a DataFrame, written with its to_csv(...)'s defaults, or a 2-D
        DataArray (year, member), written as one row per year and one
        column per member (xarray has no CSV writer of its own, so it is
        written through pandas, here)
    path:
        the CSV file to write; replaced if it exists
    year_heading:
        for a DataArray, the heading of the year column ('Year' for the
        observations, none for the piControl segments, as their files have
        always had)

    Summaries are CSV, distributions netCDF: see the output rule in
    src/outputs/distributions.py.
    """
    # A DataArray as a table: one row per year, one column per member, the
    # year column headed year_heading, the member labels unheaded
    if isinstance(table, xr.DataArray):
        table = table.to_pandas().rename_axis(index=year_heading,
                                              columns=None)
    # The whole table, written in one step
    write_atomically(path, table.to_csv)
