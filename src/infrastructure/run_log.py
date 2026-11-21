"""The run log: notes collected during a run, section headers, and the git
commit the run used.
"""

import os
import subprocess


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


def describe_git_commit():
    """The git commit of the code this run is using, for the run log.

    Recorded by gwi.py itself, at the moment it runs, rather than only by
    run-gwi.sh when it submits: a queued Slurm job reads the code on disk
    when it starts, which may have changed since it was submitted. Notes
    uncommitted changes to tracked files, which mean the commit alone does
    not describe the code. Never stops the run: without git, or outside a
    repository, it says the commit is unknown.
    """
    this_folder = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(
            ['git', '-C', this_folder, *args], capture_output=True, text=True,
            timeout=30)

    try:
        commit = git('rev-parse', '--short', 'HEAD')
        if commit.returncode != 0:
            return 'unknown (not a git repository)'
        description = commit.stdout.strip()
        # diff --quiet exits with 1 for changes (other codes are errors)
        if git('diff', '--quiet', 'HEAD').returncode == 1:
            description += ' (with uncommitted changes to tracked files)'
        return description
    except (OSError, subprocess.SubprocessError):
        return 'unknown (git not available)'
