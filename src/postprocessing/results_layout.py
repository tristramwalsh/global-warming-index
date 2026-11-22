"""Where results and figures are written.
"""
import os
import re
import uuid


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


def write_csv(df, path):
    """Write a DataFrame to a CSV file, so that no one ever sees half of it.

    df: the DataFrame, written with df.to_csv(...)'s defaults
    path: the CSV file to write; replaced if it exists

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
        df.to_csv(temp_path)
        # Move it onto the final name, in one atomic step
        os.replace(temp_path, path)
    finally:
        # Remove the temporary file if anything stopped the move
        if os.path.exists(temp_path):
            os.remove(temp_path)
