import numpy as np

from cosmostat.chi_square import (FixedParamSpec, ParamGrid,
                                  fixed_specs_as_array, fixed_specs_from_array)


def test_confidence_interval():
    """"""
    x = np.linspace(-np.pi / 2, np.pi / 2, num=129)
    y = np.sin(x) ** 2
    test_grid = ParamGrid(x, y)
    conf_interval = test_grid.get_confidence_interval(chi_square_delta=0.5)
    assert abs(conf_interval.lower_error - np.pi / 4) < 1e-3
    assert abs(conf_interval.upper_error - np.pi / 4) < 1e-3
    print(conf_interval)


def test_fixed_specs_conversion():
    """"""
    specs = [
        FixedParamSpec("param_1", 0.0),
        FixedParamSpec("param_2", 1.0),
        FixedParamSpec("param_3", -1.0),
    ]
    # Dictionary with fixed specs.
    specs_dict = {spec.name: spec.value for spec in specs}
    # Transform dict >> array >> dict
    specs_array = fixed_specs_as_array(specs_dict)
    specs_dict_new = fixed_specs_from_array(specs_array)
    # Should be equal.
    assert specs_dict == specs_dict_new
