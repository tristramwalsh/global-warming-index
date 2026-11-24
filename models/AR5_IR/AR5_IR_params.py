"""Thermal parameters for the AR5 impulse response model (AR5_IR.py), kept
for reference.

gwi.py doesn't use them: the AR5 model has no adapter in
src/method/emulators.py, so the GWI can't run with it. These were the
parameters gwi.py held for it before the FaIR v2.0 adapter became the only
emulator.

The model's defaults, of which only a[10], a[11], a[15] and a[16] matter:
    a[10:12] = [0.631, 0.429]   the AR5 thermal sensitivity coefficients
    a[15:17] = [8.400, 409.5]   the AR5 thermal time constants (Geoffroy et
                                al. give [4.1, 249.])
"""

import numpy as np

# Geoffroy et al. (2013)'s thermal time constants, for a[15:17]: the two
# time constants (rows) of each of 16 CMIP5 models (columns)
GEOFFROY_2013_TIME_CONSTANTS = np.array([
    [4.0, 5.0, 4.5, 2.8, 5.2, 3.9, 4.2, 3.6,
     1.6, 5.3, 4.0, 5.5, 3.5, 3.9, 4.3, 4.0],
    [126, 267, 193, 132, 289, 200, 317, 197,
     184, 280, 698, 286, 285, 164, 150, 218]])
