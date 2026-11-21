"""The run log: notes collected during a run, and section headers.
"""


_run_notes = []


def note(message):
    """Record something about this run that the reader should know.

    For things that change what the run does but do not stop it: a year range
    clipped to the available data, a setting that disables a safeguard, a
    resource that could not be determined. They arise in several places and
    at several stages -- argument parsing, data loading, range checking --
    and printing them where they arise would scatter them through the log
    above the first section header. Collecting them means log_notes() can put
    them together, under a heading, in a predictable place.

    Not for errors. Anything that stops the run should raise, with the detail
    in the exception message where a traceback will carry it.
    """
    _run_notes.append(message)


def log_notes():
    """Print everything note() collected, as its own section."""
    log_section('Notes and adjustments')
    if not _run_notes:
        print('None', flush=True)
        return
    for message in _run_notes:
        print(message, flush=True)


def log_section(title, width=76):
    """Start a new section of the run log.

    The log is long and is usually read after the fact, often to work out why
    a run died, so it is broken into labelled sections that can be found by
    eye or with grep.
    """
    print(f'\n── {title.upper()} '.ljust(width, '─'), flush=True)
