# PyJacAn – Automatic Analytic Jacobian Generator for Python

PyJacAn is an open-source Python package that automatically computes the **analytic Jacobian matrix** for systems of algebraic equations. This is particularly useful for solving systems with solvers like `scipy.optimize.root` that accept user-provided Jacobians, improving both **accuracy** and **computational efficiency**.

---

![Package Logo](path/to/your/logo.png)  <!-- Add your package symbol here -->

## Features

- Automatic symbolic and numerical Jacobian generation using **SymPy**.
- Supports single- and multi-equation systems, including **differential and partial differential equations**.
- Works with user-defined residual functions and boundary conditions.
- Reduces computation time and improves solver stability compared to numerical Jacobians.
- Fully compatible with Python solvers accepting analytic Jacobians.

---

## Installation

Requires **Python 3.7+**. Install dependencies via pip:

```bash
pip install numpy==1.21.5 sympy==1.10.1
```

Clone the repository:

```bash
git clone https://github.com/mc8970/Function-to-Jacobian.git
cd Function-to-Jacobian
```

---

## Usage

```python
from jacobian import jacobian
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
J_sym, J_num = jacobian(leading, var_and_lengths_dict,
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

## Dependencies

- Python 3.7+
- [NumPy](https://numpy.org/) == 1.21.5
- [SymPy](https://www.sympy.org/en/index.html) == 1.10.1

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

