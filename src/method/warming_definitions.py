"""The maths of the warming definitions: smoothing, trends and rates, and
the headline years.
"""
import numpy as np


def moving_average(data, w):
    """Calculate a moving average of data with window size w."""
    # data_padded = np.pad(data, (w//2, w-1-w//2),
    #                      mode='constant', constant_values=(0, 1.5))
    return np.convolve(data, np.ones(w), 'valid') / w


def temp_signal(data, w, method):
    """Calculate the temperature signal as moving average of window w."""
    # Sensibly extend data (to avoid shortening the length of moving average)

    # These are the lengths of the pads to add before and after the data.
    start_pad = w//2
    end_pad = w-1-w//2

    if method == 'constant':
        # Choices are:
        # - 0 before 1850 (we are defining this as preindustrial)
        # - 1.5 between 2022 and 2050 (the line through the middle)
        data_padded = np.pad(data, (start_pad, end_pad),
                             mode='constant',
                             constant_values=(0, 1.5))

    elif method == 'extrapolate':
        # Add zeros to the beginning (corresponding to pre-industrial state)
        extrap_start = np.zeros(start_pad)

        # Extrapolate the final w years to the end of the data
        A = np.vstack([np.arange(w), np.ones(w)]).T
        coef = np.linalg.lstsq(A, data[-w:], rcond=None)[0]
        B = np.vstack([np.arange(w + end_pad), np.ones(w + end_pad)]).T
        extrap_end = np.sum(coef*B, axis=1)[-end_pad:]
        data_padded = np.concatenate((extrap_start, data, extrap_end), axis=0)

    return moving_average(data_padded, w)
    return np.convolve(data_padded, np.ones(w), 'valid') / w


def _trend_weights(n):
    """Ordinary-least-squares slope weights for a fixed x-axis of 0..n-1.

    Fitting a straight line by least squares gives the slope

        b = sum_i (x_i - xbar)(y_i - ybar) / sum_i (x_i - xbar)^2

    Here x is always the integers 0..n-1, so everything involving x can be
    worked out once: with w_i = (x_i - xbar) / sum_j (x_j - xbar)^2 the slope
    is just the dot product w . y. That turns a per-series curve fit into a
    single weighted sum, which numpy can apply across a whole array at once.

    Returns (w, dx), where dx = x_last - xbar is the offset needed to evaluate
    the fitted line at its final point (see final_value_of_trend_vectorised).
    """
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    return xc / (xc @ xc), x[-1] - x.mean()


def final_value_of_trend_vectorised(array):
    """The value of the least-squares line through each series at its last
    point, along axis 0: used for the SR1.5 definition of present-day
    warming.

    Takes an array of shape (n, ...) and returns shape (...), fitting an
    independent line down axis 0 for every remaining position. The n
    points are taken to be equally spaced, x = 0, 1, ..., n - 1, and the
    line is read off at x = n - 1, the last point itself: no extrapolation
    beyond it.

    With annual means, each point stands for the middle of its year (the
    mean over 1 January to 31 December), so the value returned is the
    line's value at the middle of the final year, consistent with that
    year's annual mean. n annual means cover n years of data, and their
    first and last points are n - 1 years apart.

    The caller chooses the n years (see summaries.sr15_headline_rows and
    gwi.py's SR15_TREND_YEARS).

    The SR1.5 definition: 'warming at a given point in time is defined as
    the global average temperatures for a 30-year period centred on that
    time, extrapolating into the future if necessary'. The line's value at
    the last of n annual means equals the mean of the line extended over the
    2n - 1 annual means centred on it, n - 1 before and n - 1 after: 31
    years for n = 16, 29 for n = 15. A 30-year period centred on the middle
    of a year starts and ends mid-year, so no whole number of annual means
    gives exactly 30 years.

    A least-squares line passes through (xbar, ybar), so its value at the final
    point is ybar + b * (x_last - xbar).

    NOTE: this agrees with a per-series np.polyfit fit (the earlier
    implementation) to ~1e-7 rather than exactly. np.polyfit solves the
    least-squares problem via SVD (numpy.linalg.lstsq) instead of the closed
    form above; the two are mathematically identical but order the
    floating-point operations differently. The residual difference is float32
    rounding on values of order 1.
    """
    w, dx = _trend_weights(array.shape[0])
    return array.mean(axis=0) + np.tensordot(w, array, axes=(0, 0)) * dx


def rate_func_vectorised(array):
    """The slope of a linear trend through each series, along axis 0: used
    for the AR6 definition of the warming rate.

    Takes an array of shape (n, ...) and returns shape (...), fitting an
    independent trend down axis 0 for every remaining position and returning
    its slope.

    The slope is the dot product of the precomputed weights with the data --
    see _trend_weights for the derivation.

    NOTE: this agrees with a per-series np.polyfit fit (the earlier
    implementation) to ~1e-16, i.e. float64 machine epsilon, which is far
    tighter than final_value_of_trend_vectorised manages. The slope is a
    single weighted sum, whereas the trend endpoint additionally reconstructs
    the intercept, and that extra arithmetic on float32 inputs is what costs
    the latter its precision.
    """
    w, _ = _trend_weights(array.shape[0])
    return np.tensordot(w, array, axes=(0, 0))


def check_steps(all_reg_ranges):
    """Check that the years are in steps of 1."""
    end_yrs = sorted([
        int(regressed_years.split('-')[1])
        for regressed_years in all_reg_ranges
    ])
    all_year_steps = all(np.diff(end_yrs) == 1)

    out_dict = {
        'check_bool': all_year_steps,
        'range': f'{min(all_reg_ranges)} to {max(all_reg_ranges)}',
    }

    return out_dict


def check_headlines(hy, end_regress, end_trunc):
    """Check that the headline years are in the correct format."""

    hy = hy.replace('end_regress', str(end_regress))
    hy = hy.replace('end_trunc', str(end_trunc))
    if hy in ['IGCC', 'end_regress', 'end_trunc']:
        return hy
    elif hy.isnumeric():
        return hy
    elif all([y.isnumeric() for y in hy.split(',')]):
        return hy
    else:
        raise ValueError('Invalid headline year format.')


def generate_headline_years(headline_years, end_regress, end_trunc):
    """Generate the headline years for the analysis."""

    headline_years = headline_years.replace('end_regress', str(end_regress))
    headline_years = headline_years.replace('end_trunc', str(end_trunc))

    if headline_years == 'end_regress':
        hl_years = [end_regress]
    elif headline_years == 'end_trunc':
        hl_years = [end_trunc]

    elif headline_years == 'IGCC':
        hl_years = [end_regress]

    elif headline_years.isnumeric():
        hl_years = [int(headline_years)]

    elif all([y.isnumeric() for y in headline_years.split(',')]):
        hl_years = [int(y) for y in headline_years.split(',')]
    else:
        raise ValueError(
            f'Invalid headline year format: {headline_years!r}.')

    return hl_years


def contiguous_slice(mask):
    """Convert a boolean mask with no gaps in it into an equivalent slice.

    "No gaps" is a property of a SINGLE mask: the elements it selects must be
    adjacent to one another, so that start:stop picks out exactly the same
    set. It says nothing about how one mask relates to the next. The AR6 rate
    windows, for instance, overlap heavily from year to year (1950-1959, then
    1951-1960, ...) but each individual window is still an unbroken run and so
    converts cleanly. A mask selecting, say, indices 100, 101 and 105 could
    not, and raises.

    The headline and rate calculations select year windows with masks of the
    form (lo <= yrs) & (yrs <= hi), which by construction have no gaps.
    Indexing an array with such a mask is fancy indexing, so it COPIES the
    selected block; indexing with the equivalent slice returns a view.

    That matters at scale: on the full attribution array (~69 GB at
    samples=80, GMT-all) each mask copy costs ~1.1 s and several GB, and the
    AR6 rate loop performs one per year from 1950.

    Slicing axis 0 of a C-contiguous array yields a C-contiguous view with the
    same strides as the copy would have had, so every downstream reduction
    sees an identical memory layout and results are bitwise unchanged.

    Raises ValueError if the mask has gaps, rather than silently returning a
    slice that selects different elements.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return slice(0, 0)
    start, stop = int(idx[0]), int(idx[-1]) + 1
    if idx.size != stop - start:
        raise ValueError(
            'contiguous_slice requires a mask selecting a contiguous run; '
            f'got {idx.size} elements spanning {start}:{stop}.')
    return slice(start, stop)
