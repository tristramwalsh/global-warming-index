"""CPUs, memory and Slurm: how many workers to use, the shared results
array, threaded percentiles, and reporting what a run costs.
"""
import functools
import mmap
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import src.infrastructure.run_log as run_log
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


# lru_cache makes Python work the answer out on the first call and hand back
# that same stored number on every later call. Several of the pool sites sit
# inside loops, so without it we would fork a scontrol subprocess hundreds of
# times per run. It also guarantees the count cannot change mid-run.
@functools.lru_cache(maxsize=1)
def n_workers():
    """Return the number of CPUs actually allocated to this job.

    os.cpu_count() reports the whole node rather than the allocation, so a pool
    sized by it oversubscribes the job's memory cgroup and the workers get
    OOM-killed.

    Worker death used to be unrecoverable: mp.Pool respawns the dead worker but
    never fails the task it was holding, so map()/imap() blocked forever and the
    job burned its whole walltime at zero CPU. The pool sites now use
    ProcessPoolExecutor, which raises BrokenProcessPool instead -- but sizing
    the pool correctly is still what stops the workers being killed in the
    first place.

    No single source covers every way these scripts get run, so three are tried
    in descending order of trustworthiness:

      1. Environment variables - set by sbatch/srun job steps (gwi.py).
      2. scontrol              - interactive shells attach to step_extern,
                                 which sets none of those variables, so ask
                                 the scheduler directly (combine script).
      3. CPU affinity          - not under SLURM at all (login node, laptop).
    """
    # 1. Believe any count already stated explicitly. GWI_NUM_WORKERS is the
    #    manual override; the SLURM_* pair is set inside real job steps.
    #    CPUS_PER_TASK is preferred because CPUS_ON_NODE reports the node's
    #    whole allocation, which over-counts if a job runs >1 task per node.
    #    .get(var, '') with .isdigit() rejects missing/empty/malformed at once.
    for var in ('GWI_NUM_WORKERS', 'SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        if os.environ.get(var, '').isdigit():
            return max(1, int(os.environ[var]))

    # 2. Inside a SLURM job, but the variables above are absent - this is an
    #    interactive (step_extern) shell, so ask the scheduler itself. Note
    #    that sched_getaffinity and the cpuset cgroup are NOT restricted to the
    #    allocation on arc-htc, so they cannot stand in here. Any failure falls
    #    through to step 3 rather than killing the run.
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        try:
            output = subprocess.run(
                ['scontrol', 'show', 'job', job_id],
                capture_output=True, text=True, timeout=10).stdout
            match = re.search(r'NumCPUs=(\d+)', output)
            if match:
                return max(1, int(match.group(1)))
        except (OSError, subprocess.SubprocessError):
            pass

    # 3. No SLURM allocation at all. Prefer the CPUs this process is permitted
    #    to run on, since unlike os.cpu_count() that respects taskset/cpuset
    #    pinning where it exists. sched_getaffinity is Linux-only though, so on
    #    macOS/Windows fall back to the plain core count.
    if hasattr(os, 'sched_getaffinity'):
        n_available = len(os.sched_getaffinity(0))
    else:
        n_available = os.cpu_count() or 1

    # Finding scontrol on PATH but no job ID means we are on a cluster machine
    # outside any allocation, i.e. a login/head node, where nothing caps us and
    # the damage lands on other users. An ordinary machine sees no warning.
    if shutil.which('scontrol'):
        run_log.note(
            f'No SLURM allocation found, so falling back to {n_available} '
            f'workers. On a login/head node that spawns {n_available} '
            f'processes and degrades it for everyone else. Request an '
            f'interactive node instead (srun -p interactive -c 16 '
            f'--mem-per-cpu=2G --pty /bin/bash), or cap the workers '
            f'explicitly (GWI_NUM_WORKERS=4 python <script>.py).')

    return n_available


def shared_results_array(shape, dtype=np.float32):
    """Return an empty array that forked worker processes can write into.

    Why this exists
    ---------------
    A worker process normally has its own private memory, so anything it
    hands back to the parent has to be copied: packed up in the worker, sent
    down a pipe, and rebuilt in the parent. For a moment the same numbers
    exist twice. With one worker per FaIR parameterisation, all finishing at
    roughly the same time, that copying is the single largest memory cost of
    a GWI run. Measured on a 4 GB result: 4.0x its size when each worker
    returned its own block and the parent joined them with np.concatenate,
    3.5x when the parent wrote the returned blocks into a pre-made array,
    and 1.0x using this.

    How it works
    ------------
    The array is placed in memory the operating system marks as shareable,
    so when the parent forks a worker, the worker sees the very same memory
    rather than a private copy of it. Anything a worker writes is
    immediately visible to the parent and to every other worker. Nothing is
    sent down a pipe and workers need not return anything at all.

    Two conditions, both of which the caller must arrange:

      1. Create the array BEFORE creating the worker processes. Workers
         inherit it as they are forked; an array made afterwards is not
         shared.
      2. Start the workers with the 'fork' method, which is the Linux
         default but is worth stating explicitly, e.g.
         ProcessPoolExecutor(n, mp_context=mp.get_context('fork')).

    Each worker should write only to its own part of the array. Two workers
    writing to the same elements would race, exactly as two threads would.

    The memory holds no filename and belongs to no process in particular:
    the operating system reclaims it once the parent and all its workers
    have exited, including if they are killed outright. So there is nothing
    to clean up and nothing that can be left behind on the node.
    """
    n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    _check_and_report_memory(n_bytes, shape)
    # mmap(-1, ...) asks the OS for anonymous shared memory: no file and no
    # name. The request itself is cheap -- the pages are only charged to the
    # job as they are written -- but it can still fail outright if the array
    # is larger than the whole machine, since shareable memory is backed by
    # the node's RAM. Hence both this and the check above.
    try:
        buffer = mmap.mmap(-1, n_bytes)
    except OSError as err:
        raise MemoryError(
            f'Could not reserve {n_bytes / 1024**3:.1f} GiB for the results '
            f'of this run. That is more memory than this compute node has in '
            f'total, so no memory request would make it fit; lower the '
            f'number of samples, or use a larger node.') from err
    return np.ndarray(shape, dtype=dtype, buffer=buffer)


def _job_cgroup_value(filename):
    """Read one of Slurm's memory counters for this job, in bytes.

    Slurm confines each job to a 'cgroup', a kernel accounting box, and
    reports its memory through small files. filename is the counter's name
    under cgroup v2. Nodes running the older cgroup v1, such as ARC's, keep
    the same counters elsewhere under other names, and those are read
    instead. Returns None when not running under Slurm, or if neither can be
    read.

    In the pipeline: behind job_memory_limit and job_memory_peak, which
    gwi.py uses to report memory in its log, and to refuse up front a run
    whose results cannot fit.

    Replaces a version that read only the cgroup v2 layout, so on ARC both
    came back as None: the log showed '?' and the up-front check was
    skipped.
    """
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        return None
    v1_names = {'memory.max': 'memory.limit_in_bytes',
                'memory.peak': 'memory.max_usage_in_bytes'}
    paths = [
        f'/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{job_id}/'
        f'{filename}',
        f'/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{job_id}/'
        f'{v1_names[filename]}',
        ]
    for path in paths:
        try:
            with open(path) as fh:
                return int(fh.read())
        except (OSError, ValueError):
            continue
    return None


def job_memory_limit():
    """Return the memory this job may use, in bytes, or None outside Slurm.

    Set by --cpus-per-task x --mem-per-cpu. Exceeding it is what gets a run
    killed, so it is the number worth checking against before starting work.
    """
    return _job_cgroup_value('memory.max')


def job_memory_peak():
    """Return the most memory this job has used at once, in bytes.

    This is the kernel's own high-water mark for the job. It is a better
    figure than the MaxRSS that `sacct` reports, which adds up the memory of
    every worker process separately (so double-counts whatever they share
    with the parent) and only samples every 10 seconds (so misses brief
    peaks). This counts shared memory once and misses nothing.
    """
    return _job_cgroup_value('memory.peak')


def describe_bytes(n):
    """Format a number of bytes as GiB, or '?' if it is not known."""
    return '?' if n is None else f'{n / 1024**3:.1f} GiB'


def log_slurm_configuration():
    """Report the resources Slurm gave this job.

    Every one of these has at some point been the explanation for why two
    runs of the same configuration behaved differently: which node (their
    CPUs and memory bandwidth differ), how much memory (what gets a run
    killed), how many CPUs (the pool size), and the time limit (what turns a
    slow run into a TIMEOUT). The job ID ties the log back to `sacct` and to
    the batch directory it ends up in.
    """
    run_log.log_section('Slurm configuration')
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id is None:
        print(f'Not running under Slurm (host {os.uname().nodename})')
        return
    cpu_model = '?'
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    print(f'Job ID: {job_id}'
          f'{" (" + os.environ["SLURM_JOB_NAME"] + ")" if os.environ.get("SLURM_JOB_NAME") else ""}\n'
          f'Node: {os.uname().nodename}, partition '
          f'{os.environ.get("SLURM_JOB_PARTITION", "?")}\n'
          f'CPU: {cpu_model}\n'
          f'CPUs allocated to job: {n_workers()} '
          f'(node has {os.cpu_count()})\n'
          f'Memory allocated to job: {describe_bytes(job_memory_limit())}\n'
          f'Walltime limit: '
          f'{os.environ.get("SLURM_JOB_END_TIME") and _walltime_remaining() or "?"}',
          flush=True)


def _walltime_remaining():
    """Return the job's time limit as H:MM:SS, from Slurm's end-time stamp."""
    try:
        import datetime as _dt
        end = int(os.environ['SLURM_JOB_END_TIME'])
        start = int(os.environ.get('SLURM_JOB_START_TIME', end))
        return str(_dt.timedelta(seconds=end - start))
    except (KeyError, ValueError):
        return '?'


def _check_and_report_memory(n_bytes, shape):
    """Report what this run will cost, and stop if it cannot possibly work.

    Without this the run would start, spend minutes emulating, and only then
    be killed by the operating system with no explanation of what went wrong.
    Checking first turns that into an immediate, readable message, and
    putting the figures in the log means a run that later dies can still be
    understood from its log alone.

    The peak comes to roughly TWICE the size of the results array, because
    np.percentile copies whatever it is given (see percentile_threaded). So
    needing more than the allocation for the array alone is hopeless and
    stops the run; needing more than the allocation for twice it will almost
    certainly die later at the percentile step, which is worth saying plainly
    but not worth refusing outright, since the earlier stages may still be of
    interest.
    """
    limit = job_memory_limit()
    advice = ('lower the number of samples, or raise CPU_MEM in '
              'run-gwi.sh')

    if limit is not None and n_bytes > limit:
        raise MemoryError(
            f'This run cannot fit in the memory it has been given. The '
            f'results array alone needs {describe_bytes(n_bytes)} '
            f'({shape[0]} years x {shape[1]} variables x {shape[2]:,} '
            f'members), and this job may use only {describe_bytes(limit)}. '
            f'To fix, {advice}.')

    if limit is None:
        verdict = 'cannot tell, not running under Slurm'
    elif 2 * n_bytes > limit:
        verdict = (f'LIKELY -- the expected peak is more than the allocation, '
                   f'so this run will probably be killed at the percentile '
                   f'step. To avoid, {advice}')
    else:
        verdict = 'not expected'

    run_log.log_section('Memory usage anticipated')
    print(f'Results array: {describe_bytes(n_bytes)} '
          f'({shape[0]} years x {shape[1]} variables x {shape[2]:,} members)\n'
          f'Expected memory peak: near {describe_bytes(2 * n_bytes)} '
          f'(np.percentile copies the input from results array, doubling the '
          f'peak)\n'
          f'Slurm allocated memory for job: {describe_bytes(limit)}\n'
          f'Memory errors: {verdict}', flush=True)


def report_memory_used(results_nbytes):
    """Report what the run actually cost, to close the loop on the estimate.

    The partner of the 'Memory usage anticipated' block printed when the
    results array was made. Printing both means a log can be read on its own
    to see whether the estimate was right, without going back to Slurm's
    accounting -- which is purged eventually, and which in any case
    double-counts memory shared between worker processes.
    """
    peak, limit = job_memory_peak(), job_memory_limit()
    if peak is None:
        print('Not available (not running under Slurm)')
        return
    share = f' ({peak / limit:.0%} used)' if limit else ''
    print(f'Peak memory (usage): {describe_bytes(peak)} of '
          f'{describe_bytes(limit)} allocated{share}\n'
          f'Peak memory (vs results array): '
          f'{peak / results_nbytes:.2f}x the {describe_bytes(results_nbytes)} '
          f'results array, against the 2.00x anticipated', flush=True)


def percentile_threaded(array, q, axis, n_threads=None):
    """np.percentile, parallelised by splitting the leading axis over threads.

    numpy is not automatically parallel: it only threads operations that
    dispatch to a threaded BLAS/LAPACK (matmul, decompositions). Reductions and
    sorts -- and so np.percentile, which is built on np.partition -- are plain
    single-threaded C loops.

    Those loops do release the GIL, however, so ordinary threads parallelise
    them. Threads are used rather than processes because the attribution array
    runs to tens of GB: sharing it costs nothing, whereas mp.Pool would pickle
    a copy per worker.

    Results are bitwise identical to np.percentile(array, q, axis), because
    each leading-axis slice is reduced independently of the others and the
    per-slice computation is untouched.

    Parameters
    ----------
    array : ndarray, reduced along `axis`, split along axis 0.
    q, axis : as np.percentile. `axis` must not be 0, which is the split axis.
    n_threads : defaults to n_workers().
    """
    if axis == 0:
        raise ValueError(
            'percentile_threaded splits along axis 0, so it cannot also '
            'reduce along it; use np.percentile directly.')

    if n_threads is None:
        n_threads = n_workers()
    n_chunks = min(n_threads, array.shape[0])

    # Not worth the thread overhead, and keeps a trivially correct fallback.
    if n_chunks <= 1:
        return np.percentile(array, q, axis=axis)

    # np.percentile prepends the q axis, so the array's axis 0 becomes axis 1
    # of the result (axis != 0 is guaranteed above, so it is never consumed).
    out = None
    # Chunk with slices, NOT index arrays. array[slice] is a view, whereas
    # fancy indexing with np.array_split(np.arange(n), k) would COPY each
    # chunk -- and with every thread holding its copy at once that doubles the
    # peak memory of the whole call. On the full attribution array (~69 GB at
    # samples=80, GMT-all) that extra copy was enough to OOM a 224 GB job.
    bounds = np.linspace(0, array.shape[0], n_chunks + 1).astype(int)
    chunks = [slice(bounds[i], bounds[i + 1]) for i in range(n_chunks)]

    def _worker(sl):
        return sl, np.percentile(array[sl], q, axis=axis)

    with ThreadPoolExecutor(n_chunks) as pool:
        for sl, result in pool.map(_worker, chunks):
            if out is None:
                out = np.empty(
                    (result.shape[0], array.shape[0]) + result.shape[2:],
                    dtype=result.dtype)
            out[:, sl, ...] = result

    return out


def run_in_pool(func, tasks):
    """Call func(*task) for each task in a process pool; return the results.

    The results come back in the order of tasks. Tasks are sent in chunks
    sized so that each worker gets about four, however many tasks there are.

    In the pipeline: every parallel step of the script runs through it -- the
    averaging, the loading, and each type of base figure.

    Replaces mp.Pool, which cannot recover from a worker dying. When the
    kernel kills a worker for using too much memory, mp.Pool starts a
    replacement but never fails the task the dead worker held, so the run
    hangs; ProcessPoolExecutor raises BrokenProcessPool instead. (If it is
    this parent process the kernel kills, the shell prints 'Killed' and there
    is nothing left to report anything.)
    """
    n_processes = n_workers()
    chunksize = max(1, int(np.ceil(len(tasks) / (4 * n_processes))))
    # Frozen, the objects already in this process -- all the loaded results
    # -- are left out of the garbage collections the workers make after each
    # figure (close_figure), which then stay quick and do not copy the
    # parent's memory into every worker.
    gc.collect()
    gc.freeze()
    # The workers must be forked, not spawned. A forked worker starts as a
    # copy of this process, with the output folders set_output_folders() set
    # and the results already loaded; a spawned one starts afresh and has
    # neither. Fork is the default on Linux but not on macOS (spawn, since
    # Python 3.8), so it is requested explicitly, as gwi.py does.
    try:
        with ProcessPoolExecutor(
                n_processes, mp_context=mp.get_context('fork')) as executor:
            return list(executor.map(func, *zip(*tasks), chunksize=chunksize))
    finally:
        gc.unfreeze()
