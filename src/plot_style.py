"""The look of every figure: the matplotlib settings (fonts, axes, grid)
and the colours and line styles of the variables and headlines.

combine_results_iterations.py applies the settings with plot_style.use(),
before it draws anything.
"""
import matplotlib

# font_family = 'Roboto'
font_family = 'Lato'

RC_PARAMS = {
    # General figure
    'figure.dpi': 300,
    'figure.figsize': (15, 10),
    'figure.titlesize': 17,
    'figure.titleweight': 'light',
    'legend.frameon': False,
    # General fonts
    'font.family':  font_family,
    'font.weight': 'light',
    'font.size': 11,
    'pdf.fonttype': 42,  # Switch from default 3 to 42 to use TrueType fonts
    # for published PDF figures
    # Mathtext fonts
    'mathtext.fontset': 'custom',
    'mathtext.rm': font_family,
    'mathtext.bf': f'{font_family}:bold',
    'mathtext.cal': font_family,  # To pre-emptively stop the matplotlib error of
    # being unable to find a calligraphic/cursive font on the linux cluster,
    # specigy to just use font_family font choice for these cases.
    # Axis box
    'axes.spines.bottom': True,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.5,
    # 'axes.facecolor': '#f9f8f7',  # AR6 at 50% opacity.
    'axes.facecolor': 'white',
    # Axis labels
    'axes.titleweight': 'regular',
    # 'axes.labelcolor': 'gray',
    # Axis grid
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#cfd1d0',  # AR6
    'grid.linewidth': 0.5,
    # 'axes.axisbelow': True,
    # Axis  ticks
    'ytick.major.size': 0,
    'ytick.major.width': 0,
    # 'ytick.color': 'gray',
}
# Fontweights~: light, regular, normal,


def use():
    """Apply the figure settings (RC_PARAMS) to matplotlib."""
    matplotlib.rcParams.update(RC_PARAMS)


# Constants

VAR_COLOURS = {
    'Tot': '#d7827e',
    'Ant': '#b4637a',
    'GHG': '#907aa9',
    'Nat': '#56949f',
    'OHF': '#ea9d34',
    'Res': '#9893a5',
    'Obs': '#797593',
    'PiC': '#cecacd'
}

HEADLINE_COLOURS = {
    'ANNUAL': '#5BA2D0',
    'SR15': '#9CCFD8',
    'AR6': '#EE8679',
    'CGWL': '#A88BFA'
}
HEADLINE_LINE_STYLE = {
    'Tot': 'solid',
    'Ant': 'dashed',
    'Nat': 'dotted'
}
