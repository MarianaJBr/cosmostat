import numpy as np
from chisquarecosmo.chi_square import ParamGrid


def test_confidence_interval():
    """"""
    x = np.linspace(-np.pi / 2, np.pi / 2, num=129)
    y = np.sin(x) ** 2
    test_grid = ParamGrid(x, y)
    conf_interval = test_grid.get_confidence_interval(chi_square_delta=0.5)
    assert abs(conf_interval.lower_error - np.pi / 4) < 1e-3
    assert abs(conf_interval.upper_error - np.pi / 4) < 1e-3
    print(conf_interval)
