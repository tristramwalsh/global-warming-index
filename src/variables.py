"""The forcing variables: their names, how they aggregate, and which
variables a regression diagnoses.
"""


SUB_VAR_MAPPING = {
    'GHG': ['co2', 'ch4', 'n2o', 'halogen'],
    'OHF': ['aerosol-radiation_interactions', 'aerosol-cloud_interactions',
            'contrails', 'land_use', 'bc_snow', 'h2o_strat', 'o3'],
    'Nat': ['solar', 'volcanic'],
    'Ant': ['GHG', 'OHF'],
    'Tot': ['Ant', 'Nat']
}


VAR_NAMES = {
    'Obs': 'Observed warming',
    'Tot': 'Total forced warming',
    'Ant': 'Human-induced warming',
    'GHG': 'Well-mixed greenhouse gases',
    'OHF': 'Other human forcings',
    'Nat': 'Solar and volcanic drivers',
    'Res': 'Residual (Internal variability)',
    'co2': 'Carbon dioxide',
    'ch4': 'Methane',
    'n2o': 'Nitrous oxide',
    'halogen': 'Halogenated gases',
    'aerosol-radiation_interactions': 'Aerosol-radiation interactions',
    'aerosol-cloud_interactions': 'Aerosol-cloud interactions',
    'land_use': 'Land-use reflectance',
    'bc_snow': 'Black carbon on snow',
    'h2o_strat': 'Stratospheric water vapour',
    'o3': 'Ozone',
    'solar': 'Solar',
    'volcanic': 'Volcanic',
    'contrails': 'Aviation contrails'
}


def extra_vars(forc_vars):
    r"""Return diagnosable variables for a set of regressed variables."""
    # 1-way regression of Tot against Obs
    if 'Tot' in forc_vars and len(forc_vars) == 1:
        extra_vars = ['Res']

    # 2-way regression of Ant and Nat against Obs
    elif 'Ant' in forc_vars and 'Nat' in forc_vars and len(forc_vars) == 2:
        extra_vars = ['Tot', 'Res']

    # 3-way regression
    elif 'Ant' not in forc_vars and len(forc_vars) == 3:
        extra_vars = ['Ant', 'Tot', 'Res']

    else:
        # Raise error:
        raise ValueError('Invalid combination of variables for regression.')

    return extra_vars


def map_var_to_regression_aggregate(var_list_ERF, regress_vars):
    """
    Creates a mapping from each variable in var_list_ERF to its corresponding
    regression variable in regress_vars.
    """

    # First, remove the extra variables that are a higher-level combination of
    # the regression variables, so that the recursive search only finds the
    # correct regression variable, and not anything "above it" which will be
    # calculated as a linear combination later.
    extra_vars_list = extra_vars(regress_vars)
    reduced_mapping = {k: v for k, v in SUB_VAR_MAPPING.items()
                       if k not in extra_vars_list}

    def get_highest_parent(target, mapping):
        for parent, children in mapping.items():
            if target in children:
                return get_highest_parent(parent, mapping)
        return target

    # Explicitly ensure regression variables map to themselves
    # First, turn the inverse map function above into a dictionary map
    mapped = {}
    for v in var_list_ERF:
        mapped[v] = get_highest_parent(v, reduced_mapping)
    # Second, apply the identity mapping for regression variables
    for rv in regress_vars:
        mapped[rv] = rv

    return mapped
