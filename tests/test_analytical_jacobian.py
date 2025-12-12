# test_example.py

# Important!: Name file with test_ prefix so pytest can find it
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
#sys.path.append("../src")  # to find pyjacan module one folder up
#sys.path.append("../src/pyjacan")  # to find pyjacan module one folder up

import numpy as np
import sympy as sym
from pyjacan.core import analytical_jacobian



def bulk(var, i, vars_):
    """
    Simple 2 function vector:
        f1(x, y) = x*y
        f2(x, y) = sin(x) + 3*y**2
    """
    x = vars_["x"][0]  # symbol x0
    y = vars_["y"][0]  # symbol y0

    if var == "x":
        # First equation: f1(x, y) = x*y
        return y * x
    elif var == "y":
        # Second equation: f2(x, y) = sin(x) + 3*y**2
        return sym.sin(x) + 3 * y**2
    else:
        raise ValueError(f"Unexpected variable name {var!r}")


def BC(vars_):
    """No BC for simple Jacobian test."""
    return []


def test_jacobianc():
    # One DOF for each variable x and y
    var_and_length_dict = {"x": 1, "y": 1}

    # No “cut” at left/right (no boundary nodes removed)
    lr_cut = [
        {"x": 0, "y": 0},  # left
        {"x": 0, "y": 0},  # right
    ]

    # Evaluation point: (x, y) = (0, 2)
    x_val = 0.0
    y_val = 2.0

    # Internal symbols created by pyJacAn are 'x0' and 'y0'
    values_dict = {"x0": x_val, "y0": y_val}

    # Get symbolic and numeric Jacobian from PyJacAn
    J_sym, J_num = analytical_jacobian(
        bulk,
        var_and_length_dict,
        BC,      # left_boundary
        BC,      # right_boundary
        lr_cut,
        values_dict,
    )

    # Expected analytic Jacobian:
    #   J(x, y) = [[cos(x), 6*y   ],
    #              [y,      x   ]]
    # evaluated at (x, y) = (0, 2):
    #   J(0, 2) = [[1, 12],
    #              [2, 0]]
    J_expected = np.array(
        [
            [np.cos(x_val), 6.0 * y_val],
            [y_val, x_val],
        ],
        dtype=float,
    )

    # Converting numerical Jacobian to array to be asserted
    J_pyjacan = np.array(J_num, dtype=float)
    
    # Asserting with pytest
    assert np.allclose(J_pyjacan, J_expected)

    # Run in terminal in same folder as file as: python -m pytest -q