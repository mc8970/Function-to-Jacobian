#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sympy as sym
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from Module.src.main import analytical_jacobian

# -----------------------------------------------------------------------------
# Definition of bulk (differential scheme) and boundary functions for diffusion
# -----------------------------------------------------------------------------

def bulk(var, i, vars):
    """
    Defines the bulk equation for the diffusion problem.
    var : str, variable name (here only 'u')
    i   : int, spatial index
    vars: dict, symbolic variables (Sympy)
    """
    u = vars['u']
    D, dt, dx, u_old = sym.symbols('D dt dx u_old')
    if var == 'u':
        # Discretized diffusion scheme (central difference in space, Euler in time)
        return D * (u[i - 1] - 2 * u[i] + u[i + 1]) * dt / dx**2 + u_old - u[i]

def left_boundary(vars):
    """
    Left boundary condition (Neumann type, derivative = 0 at left boundary).
    """
    u0 = vars['u'][0]
    u1 = vars['u'][1]
    dx = sym.symbols('dx')
    return [(u1 - u0) / dx**2]

def right_boundary(vars):
    """
    Right boundary condition (Neumann type, derivative = 0 at right boundary).
    """
    u_n = vars['u'][-1]
    u_nn = vars['u'][-2]
    dx = sym.symbols('dx')
    return [(-u_n + u_nn) / dx**2]

# -----------------------------------------------------------------------------
# Problem setup: variables, boundaries, and parameter values
# -----------------------------------------------------------------------------
var_and_lengths_dict = {'u': 5}                       # number of spatial points for variable 'u'
lr_cut = [{'u': 1}, {'u': 1}]                         # left and right cuts (exclude boundary points)
values_dict = {'D': 0.5, 'dt': 0.5, 'dx': 10/4}       # parameters

# Generate the Jacobian (symbolic + numeric evaluation)
jacobian = analytical_jacobian(bulk, var_and_lengths_dict, left_boundary, right_boundary, lr_cut, values_dict)

# Output symbolic Jacobian (jacobian[0]) or evaluated numeric Jacobian (jacobian[1])
jacobian[0]
# jacobian[1]


# In[2]:

# -----------------------------------------------------------------------------
# Function definition for root solver (nonlinear system)
# -----------------------------------------------------------------------------
def fun(u):
    """
    Constructs the discretized diffusion equation system for solver.
    Includes left boundary, bulk, and right boundary equations.
    """
    u_xx = np.zeros(Nx)
    u_xx[0] = (u[1] - u[0]) / dx**2           # left boundary condition
    u_xx[-1] = (- u[-1] + u[-2]) / dx**2      # right boundary condition
    for i in range(1, Nx - 1):                # bulk scheme (central differences)
        u_xx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
    return u - u0 - dt * D * u_xx             # equation form F(u) = 0


# In[3]:

import timeit

# -----------------------------------------------------------------------------
# Problem parameters and initial condition
# -----------------------------------------------------------------------------
L = 10         # Length of the domain
D = 0.5        # Diffusion coefficient
T = 100        # Total simulation time
Nx = 5         # Number of spatial discretization points
dt = 0.5       # Time step

dx = L / (Nx - 1)                             # spatial step
x = np.linspace(0, L, Nx, endpoint=True)      # spatial coordinates

A = 10.0       # Peak amplitude of Gaussian
mu = 5         # Center of Gaussian
sigma = 0.25   # Standard deviation

# Initial Gaussian profile
initial_u0 = A * np.exp(-0.5 * ((x - mu) / sigma)**2)
u0 = A * np.exp(-0.5 * ((x - mu) / sigma)**2)

def jac(u):
    """
    Returns numeric Jacobian matrix (precomputed from analytical_jacobian).
    """
    return jacobian[1]


# In[4]:

# -----------------------------------------------------------------------------
# Root solver with or without Jacobian
# -----------------------------------------------------------------------------
def solution_root(our_jac, u0):
    """
    Solves the nonlinear system using root solver.
    our_jac : bool or function, if True uses Jacobian, otherwise not
    u0      : initial condition
    """
    t = 0
    solution = [u0.copy()]   # save solutions at each step
    times = [t]              # save times

    while t < T:
        if our_jac:
            sol = root(fun, u0, jac=jac)      # with Jacobian
        else:
            sol = root(fun, u0)               # without Jacobian
        u0 = sol.x
        t += dt
        solution.append(u0.copy())
        times.append(t)
    
    return solution, times


# In[5]:

# -----------------------------------------------------------------------------
# Timing comparison: with and without Jacobian
# -----------------------------------------------------------------------------
time_without_jac = timeit.timeit(lambda: solution_root(our_jac=False, u0=initial_u0.copy()), number=2)
time_with_jac = timeit.timeit(lambda: solution_root(our_jac=jac, u0=initial_u0.copy()), number=2)

print(f"Time without Jacobian matrix argument: {time_without_jac:.4f} [s]")
print(f"Time with Jacobian Matrix argument: {time_with_jac:.4f} [s]")
print(f"The difference is {(time_with_jac - time_without_jac):.4f} [s] or {((time_with_jac - time_without_jac)/time_without_jac)*100:.4f}% faster/slower (- faster with jac, + slower with jac)")


# In[6]:

# -----------------------------------------------------------------------------
# Plotting an example of performance comparison (already hardcoded values)
# -----------------------------------------------------------------------------
x1 = [0, 5, 10, 50, 100]
ybz = [0, 0.0373, 0.0714, 0.7377, 2.3704]  # no Jacobian
yz = [0, 0.024, 0.03, 0.2882, 0.9147]      # with Jacobian
yproc = [0, 27.59, 58.5, 60.9, 61.4]       # percentage savings

plt.plot(x1, yproc)
plt.ylim(0, 100)
plt.xlabel(f'Number of points')
plt.ylabel(f'Decrease in execution time [%]')
plt.grid(True)
plt.show()


# In[7]:

# -----------------------------------------------------------------------------
# Comparison of numerical solutions with and without Jacobian
# -----------------------------------------------------------------------------
solution_without_jac, _ = solution_root(False, initial_u0.copy())
solution_with_jac, _ = solution_root(True, initial_u0.copy())

# Compare final results
difference = np.linalg.norm(solution_without_jac[-1] - solution_with_jac[-1])
print(f"The difference in final results: {difference:.4e}")

# Compare entire trajectories
entire_difference = 0
for i, j in zip(solution_without_jac, solution_with_jac):
    entire_difference += np.linalg.norm(abs(i - j))

print(f"The entire difference: {entire_difference:.4e}")


# In[8]:

# -----------------------------------------------------------------------------
# Analytical solution for comparison
# -----------------------------------------------------------------------------
from scipy.integrate import simpson as simps
from matplotlib.animation import FuncAnimation

L = 10
D = 0.5
T = 100
points = 100000 + 1
dt = 1
anal_dx = L / (points - 1)
anal_x = np.linspace(0, L, points, endpoint=True)

A = 10.0
mu = L / 2
sigma = 0.25

# Initial condition (analytical grid)
anal_init_u0 = A * np.exp(-0.5 * ((anal_x - mu) / sigma)**2)

def analytical_solution(x, t, L, D, suma, points):
    """
    Analytical Fourier series solution for 1D diffusion equation.
    suma   : number of terms in Fourier series
    points : number of discretization points in space
    """
    equation = 0
    point_array = np.linspace(0, L, points, endpoint=True)
    u0 = 10.0 * np.exp(-0.5 * ((x - mu) / sigma)**2)
    C_0 = simps(y=u0, x=point_array) / L
    for n in range(1, suma + 1):
        cos_component = np.cos(n * np.pi * point_array / L)
        integrand = u0 * cos_component
        C_n = 2 * simps(y=integrand, x=point_array) / L
        en = C_n * np.cos(n * np.pi * x / L) * np.exp(-D * (n * np.pi / L) ** 2 * t)
        equation += en
    equation += C_0
    return equation

# Iteratively determine required number of Fourier terms
limit = 10**-12
number_of_components = 95
difference = np.inf

while difference > limit:
    solution = analytical_solution(anal_x, 0, L, D, number_of_components, points)
    difference = np.max(np.abs(solution - anal_init_u0))
    print(f"Iteration: {number_of_components}")
    print(f"Difference: {difference:.12f}")
    number_of_components += 1

print(f"Converges with the following number of components : {number_of_components}")

# Plot analytical solution vs initial condition
solution = analytical_solution(anal_x, 0.1, L, D, 95, points)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(anal_x, anal_init_u0, label='Gaussian initial condition')
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('u(x,t) [Number of Particles]')
axs[0].set_title('Initial condition')
axs[0].legend(loc='upper right')
axs[0].grid(True)

axs[1].plot(anal_x, anal_init_u0, label='Gaussian initial condition')
axs[1].plot(anal_x, solution, label=f'Analytical solution with {number_of_components} terms')
axs[1].set_xlabel('x [m]')
axs[1].set_ylabel('u(x,t) [Number of Particles]')
axs[1].set_title('Best analytical solution at t = 0.1 s')
axs[1].legend(bbox_to_anchor=(1, 1))
axs[1].grid(True)
plt.tight_layout()
plt.show()

# Values at specific points
x_val = [0, 2.5, 5, 7.5, 10]
specific_solutions = {}
for x in x_val:
    index = np.where(np.isclose(anal_x, x))[0][0]
    specific_solutions[x] = solution[index]

for x, val in specific_solutions.items():
    print(f"Value at x = {x}: {val}")

array_solutions = np.array(list(specific_solutions.values()))


# In[9]:

# -----------------------------------------------------------------------------
# Comparison of numerical and analytical solutions at specific times
# -----------------------------------------------------------------------------
time = 10
dt = 0.5
index = int(time / dt)

solution = analytical_solution(anal_x, time, L, D, number_of_components, points)
x_val = [0, 2.5, 5, 7.5, 10]

with_jac = 0
without_jac = 0
t = np.linspace(0, time, 10, endpoint=True)

# Compute deviations over time
for t_i in t:
    t_index = int(t_i/dt)
    anal_solution = analytical_solution(anal_x, t_i, L, D, number_of_components, points)
    specific_solutions = {}
    for x in x_val:
        ind = np.where(np.isclose(anal_x, x))[0][0]
        specific_solutions[x] = anal_solution[ind]

    array_solutions = np.array(list(specific_solutions.values()))
    for i, j, k in zip(solution_with_jac[t_index], solution_without_jac[t_index], array_solutions):
        with_jac += np.linalg.norm(abs(i - k))
        without_jac += np.linalg.norm(abs(j - k))

print(f'Jac {with_jac}')
print(f'Without jac {without_jac}')
print(f'Difference {with_jac-without_jac} + bigger with jac, - bigger without jac or {100*((with_jac-without_jac)/without_jac)}%')

print(f'With jac values: {solution_with_jac[index]}')
print(f'Without jac values: {solution_without_jac[index]}')
print(f'Analytical solutions at t = {time} are: {array_solutions}')

# Plotting comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(anal_x, solution, label=f'Analytical solution')
axs[0].scatter(x_val, solution_with_jac[index], label=f'Numerical solution', color='red')
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('u(x,t) [Number of Particles]')
axs[0].set_title(f'Solution without Jacobian matrix at t = {time} s')
axs[0].legend(loc='best')
axs[0].grid(True)

axs[1].plot(anal_x, solution, label=f'Analytical solution')
axs[1].scatter(x_val, solution_without_jac[index], label=f'Numerical solution', color='black')
axs[1].set_xlabel('x [m]')
axs[1].set_ylabel('u(x,t) [Number of Particles]')
axs[1].set_title(f'Solution with Jacobian Matrix at t = {time} s')
axs[1].legend(loc='best')
axs[1].grid(True)
plt.show()
