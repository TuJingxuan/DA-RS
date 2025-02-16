import daceypy_import_helper  # noqa: F401
from typing import Callable, Type
import numpy as np
import math
import scipy
from daceypy import DA, RK, array, integrator, ADS

def envelope_equation_partial_map_inversion(
        envelope_equation: array,
        point: np.array,
) -> array:
    """Perform partial map inversion for the envelope equation"""
    """Generate a map based on the given point"""
    theta = point[0] + DA(1)
    r = point[1] + DA(2)
    point_input = array([
        r * theta.cos(), r * theta.sin(), 0,
    ])
    """Perform partial map inversion"""
    AugMap = array([
        DA(1),
        envelope_equation.eval(point_input),
        DA(3),
    ])
    ParInvMap = AugMap.invert()
    ParInvMap = ParInvMap.plug(2, 0).plug(3, 0)[1]
    """Return results"""
    return ParInvMap

