"""Where results and figures are written.
"""
import re


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
