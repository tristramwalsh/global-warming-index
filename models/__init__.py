"""Climate models used to emulate the temperature response to forcing.

Each model has its own folder here, holding the model's code or parameters
and a SOURCE.md; the GWI reaches them only through src/method/emulators.py:
- AR5_IR/:  the IPCC AR5 impulse response model and its default parameters
            (not wired into the current analysis).
- FaIR_V2/: the CMIP6 climate-response parameters for FaIR v2.0.0-alpha1; the
            FaIR code itself is downloaded into FaIR_V2/FaIRv2_0_0_alpha1/ the
            first time this package is imported, and is not tracked by git.
"""

import os
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


# Paths are relative to this file, so they don't depend on the working
# directory.
FAIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FaIR_V2')
FAIR_CODE_DIR = os.path.join(FAIR_DIR, 'FaIRv2_0_0_alpha1')

# FaIR_V2/ is tracked (it holds SOURCE.md and the CMIP6 parameters), so it
# always exists: check for the downloaded code, not the folder.
if not os.path.isdir(FAIR_CODE_DIR):
    print('FaIR does not exist - downloading now.')
    os.makedirs(FAIR_DIR, exist_ok=True)
    download_and_unzip(
        url='https://zenodo.org/record/4774994/files/FaIRv2.0.0-alpha1.zip',
        extract_to=FAIR_DIR
    )
    os.rename(os.path.join(FAIR_DIR, 'FaIRv2.0.0-alpha1'), FAIR_CODE_DIR)

    for f in os.listdir(FAIR_CODE_DIR):
        if 'fair' not in f and 'notebooks' not in f:
            os.remove(os.path.join(FAIR_CODE_DIR, f))

    # Prevent tqdm visualisation from running in FaIR.
    with open(os.path.join(FAIR_CODE_DIR, 'fair', 'fair_runner.py'),
              'r+') as file:
        filedata = file.read()
        filedata = filedata.replace(
            "tqdm(np.arange(1,n_year),unit=' timestep')",
            "np.arange(1,n_year)"
        )
        # Overwrite the file rather than appending a second copy to it.
        file.seek(0)
        file.write(filedata)
        file.truncate()
