# PyJacAn – Automatic Analytic Jacobian Generator for Python

PyJacAn is an open-source Python package that automatically computes the **analytic Jacobian matrix** for systems of algebraic equations. This is particularly useful for solving systems with solvers like `scipy.optimize.root` that accept user-provided Jacobians, improving both **accuracy** and **computational efficiency**.

---

<!-- PROJECT LOGO -->
<br />

<div align="center">
    <img src="Logo.png" alt="logo" width="250" height="300">
</div>

## Features

- Automatic symbolic and numerical Jacobian generation using **SymPy**.
- Supports single- and multi-equation systems, including **differential and partial differential equations**.
- Works with user-defined residual functions and boundary conditions.
- Reduces computation time and improves solver stability compared to numerical Jacobians.
- Fully compatible with Python solvers accepting analytic Jacobians.

---

## Installation

You can install PyJacAn in two ways:

### 1. Install via pip (recommended)
PyJacAn is pip-installable. Run:

```bash
pip install pyjacan
```
This will automatically install all dependencies:
- numpy>=1.24
- sympy>=1.12
- scipy>=1.11
- matplotlib>=3.8


### 2. Manual installation from GitHub
You can also clone the repository and install manually:

```bash
git clone https://github.com/mc8970/Function-to-Jacobian.git
cd Function-to-Jacobian
pip install -e .
```

This method also installs all required dependencies automatically.

---

## Usage

```python
from pyjacan import analytical_jacobian
import sympy as sym

# Define your residual functions
def leading(var, i, vars):
    # Example for coupled reaction-diffusion
    C, T = vars['C'], vars['T']
    dt, dx, D, alpha, gamma1, gamma2 = sym.symbols('dt dx D alpha gamma1 gamma2')
    if var == 'C':
        return C[i] + dt*(D*(C[i-1]-2*C[i]+C[i+1])/dx**2 - gamma2*T[i])
    elif var == 'T':
        return T[i] + dt*(alpha*(T[i-1]-2*T[i]+T[i+1])/dx**2 + gamma1*C[i])

# Define boundary conditions
def left_boundary(vars):
    return [vars['C'][0]-0.01, vars['T'][0]-293]

def right_boundary(vars):
    dx = sym.Symbol('dx')
    return [(vars['C'][-1]-vars['C'][-2])/dx, (vars['T'][-1]-vars['T'][-2])/dx]

# Define variables
var_and_lengths_dict = {'C':5, 'T':5}
lr_cut = [{'C':1,'T':1}, {'C':1,'T':1}]
values_dict = {'D':1e-9, 'alpha':1e-6, 'gamma1':0.05, 'gamma2':0.01, 'dt':0.1, 'dx':0.2}

# Generate Jacobian
J_sym, J_num = analytical_jacobian(leading, var_and_lengths_dict,
                        left_boundary, right_boundary,
                        lr_cut, values_dict)
```

---

## Validation

PyJacAn has been tested with:

1. **Diffusion equation** – verified against analytical Fourier series solutions.
2. **Coupled reaction–diffusion systems** – multi-equation systems with mixed boundary conditions.
3. **Transient heat conduction** – verified using automated **Pytest** unit tests.

Tests confirm that Jacobians are **accurate, stable, and solver-ready**, reducing computation time while preserving solution quality.

---

## Running Tests

PyJacAn includes **automated unit tests** to ensure that the analytic Jacobian calculations are correct. These tests are implemented using **Pytest**.

### Running all tests

From the project root directory, run:

python -m pytest -q

This will automatically discover all test files in the `tests/` folder and report the results. For example, a passing test will show:

1 passed in 0.52s

### Running a specific test file

To run only the test for the analytic Jacobian:

python -m pytest tests/test_analytical_jacobian.py -q

### Notes

- Tests are automatically set up to work with the **src/pyjacan** layout.  
- No additional configuration is needed if PyJacAn was installed via `pip install -e .` or dependencies were installed.  
- If a test fails, Pytest will show the failed assertion and the expected vs. computed values for easy debugging.

---

## Dependencies

- [Python](https://python.org/) >= 3.7+
- [NumPy](https://numpy.org/) == 1.21.5
- [SymPy](https://www.sympy.org/en/index.html) == 1.10.1
- [SciPy](https://scipy.org/) >= 1.11
- [Matplotlib](https://matplotlib.org/) >= 3.8
---

## License

- Code repository: [GPLv3](https://github.com/mc8970/Function-to-Jacobian)  
- Zenodo archive: [DOI:10.5281/zenodo.14056930](https://doi.org/10.5281/zenodo.14056930)

---

## Citation

If you use PyJacAn in your research, please cite:

> Jeraj, N. PyJacAn: Automatic Analytic Jacobian Generator, Zenodo, 2024. DOI: [10.5281/zenodo.14056930](https://doi.org/10.5281/zenodo.14056930)

---

## Contact

For questions or feedback, contact the author via the GitHub repository.

