import sympy as sym
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import timeit
from scipy.integrate import simpson

#If you have installed the package via pip, use the following line
#from pyjacan import analytical_jacobian

#if you are running the code from the source folder, use the following line instead
from src.pyjacan.core import analytical_jacobian


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
    D, dt, dx, uold = sym.symbols('D dt dx uold')
    if var == 'u':
        # Discretized diffusion scheme (central difference in space, Euler in time)
        return D * (u[i - 1] - 2 * u[i] + u[i + 1]) * dt / dx**2 + uold - u[i]

def left_boundary(vars):
    """
    Left boundary condition (Neumann type, derivative = 0 at left boundary).
    """
    u_0 = vars['u'][0]
    u1 = vars['u'][1]
    #dx = sym.symbols('dx')
    D, dt, dx, uold = sym.symbols('D dt dx uold')
    return [D * (u1 - u_0) *dt / dx**2 + uold - u_0]

def right_boundary(vars):
    """
    Right boundary condition (Neumann type, derivative = 0 at right boundary).
    """
    u_n = vars['u'][-1]
    u_nn = vars['u'][-2]
    #dx = sym.symbols('dx')
    D, dt, dx, uold = sym.symbols('D dt dx uold')
    return [D * (-u_n + u_nn) * dt / dx**2 + uold - u_n]



# -----------------------------------------------------------------------------
# Function definition for root solver (nonlinear system)
# -----------------------------------------------------------------------------
def fun(u, u_old):
    """
    Constructs the discretized diffusion equation system for solver.
    Includes left boundary, bulk, and right boundary equations.
    """
    u_xx = np.zeros(Nx)
    u_xx[0] = (u[1] - u[0]) / dx**2           # left boundary condition
    u_xx[-1] = (- u[-1] + u[-2]) / dx**2      # right boundary condition
    for i in range(1, Nx - 1):                # bulk scheme (central differences)
        u_xx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
    return u - u_old - dt * D * u_xx             # equation form F(u) = 0


def jac(u):
    """
    Returns numeric Jacobian matrix (precomputed from analytical_jacobian).
    """
    return jacobian[1]

# -----------------------------------------------------------------------------
# Root solver with or without Jacobian
# -----------------------------------------------------------------------------
def solution_root(our_jac, u0):
    """
    Solves the nonlinear system using root solver.
    our_jac : bool or function, if True uses Jacobian, otherwise not
    u0      : initial condition
    """
    solution = [u0.copy()]   # save solutions at each step
    times = [0]              # save times
    u_old = u0

    for i in range(0, int(T/dt)):
        if our_jac:
            sol = root(lambda u: fun(u, u_old), u_old, jac=jac)      # with Jacobian
        else:
            sol = root(lambda u: fun(u, u_old), u_old)               # without Jacobian
        #print(sol.x)
        u_old = np.array(sol.x)
        solution.append(sol.x)
        times.append(i*dt)
    
    return solution, times

def analytical_solution_on_grid(x_eval, t, L, D, u0, x_grid, suma=150, points=20001):
    """
    Compute analytical Fourier cosine series solution on x_eval grid.
    
    x_eval  : points where solution is evaluated
    t       : time
    L       : domain length
    D       : diffusion coefficient
    u0      : initial condition array
    x_grid  : spatial grid corresponding to u0
    suma    : number of Fourier cosine terms
    points  : number of points for dense integration
    """
    # Dense integration grid for numerical integration
    dense_x = np.linspace(0, L, points)
    # Interpolate initial condition to dense grid
    #u0_dense = np.interp(dense_x, x_grid, u0)
    u0_dense = A * np.exp(-0.5 * ((dense_x - mu) / sigma)**2)

    # Zeroth cosine coefficient
    C0 = simpson(u0_dense, dense_x) / L

    # Build dense solution
    sol_dense = np.full(points, C0)
    for n in range(1, suma + 1):
        cos_component = np.cos(n * np.pi * dense_x / L)
        Cn = 2.0 * simpson(u0_dense * cos_component, dense_x) / L
        sol_dense += Cn * cos_component * np.exp(-D * (n * np.pi / L)**2 * t)

    # Interpolate to x_eval
    return np.interp(x_eval, dense_x, sol_dense)

# -----------------------------------------------------------------------------
# Problem parameters and initial condition
# -----------------------------------------------------------------------------
L = 10         # Length of the domain
D = 0.5        # Diffusion coefficient
T = 50          # Total simulation time
Nx = 231         # Number of spatial discretization points
dt = 0.1       # Time step
N_iter = int(T/dt)  # Number of time steps

dx = L / (Nx - 1)                             # spatial step
x = np.linspace(0, L, Nx, endpoint=True)      # spatial coordinates

A = 10.0       # Peak amplitude of Gaussian
mu = 5         # Center of Gaussian
sigma = 0.25   # Standard deviation

# Initial Gaussian profile
u0 = A * np.exp(-0.5 * ((x - mu) / sigma)**2)


var_and_lengths_dict = {'u': Nx}                       # number of spatial points for variable 'u'
lr_cut = [{'u': 1}, {'u': 1}]                         # left and right cuts (exclude boundary points)
values_dict = {'D': D, 'dt': dt, 'dx': dx}       # parameters

# Generate the Jacobian (symbolic + numeric evaluation)
jacobian = analytical_jacobian(bulk, var_and_lengths_dict, left_boundary, right_boundary, lr_cut, values_dict)

# -----------------------------------------------------------------------------
# Comparison of computational times with and without Jacobian
# -----------------------------------------------------------------------------

time_without_jac = timeit.timeit(lambda: solution_root(our_jac=False, u0=u0.copy()), number=2)
time_with_jac = timeit.timeit(lambda: solution_root(our_jac=jac, u0=u0.copy()), number=2)


print(f"Time without Jacobian matrix argument: {time_without_jac:.4f} [s]")
print(f"Time with Jacobian Matrix argument: {time_with_jac:.4f} [s]")
print(f"The difference is {(time_with_jac - time_without_jac):.4f} [s] or {((time_with_jac - time_without_jac)/time_without_jac)*100:.4f}% faster/slower (- faster with jac, + slower with jac)")


# -----------------------------------------------------------------------------
# Comparison of numerical solutions with and without Jacobian
# -----------------------------------------------------------------------------
solution_without_jac, _ = solution_root(False, u0.copy())
solution_with_jac, _ = solution_root(True, u0.copy())

# Compare final results
difference = np.linalg.norm(solution_without_jac[-1] - solution_with_jac[-1])
print(f"The difference in final results: {difference:.4e}")

plt.figure(figsize=(8, 5))
plt.plot(x, solution_without_jac[-1], label='Without Analytic Jacobian', marker='o')
plt.plot(x, solution_with_jac[-1], label='With Analytic Jacobian', marker='x')
plt.xlabel('x')
plt.ylabel('u(x, t=T)')
plt.title('Comparison of Numerical Solutions at Final Time')
plt.grid(True)
plt.legend()
plt.show()

# --------------------------------------------------------
# Compare initial condition
# --------------------------------------------------------
#anal_init = analytical_solution_on_grid(x, t=0.0, L=L, D=D, u0=u0, x_grid=x, suma=200)

#plt.figure(figsize=(6,4))
#plt.plot(x, u0, 'k-', label='Original Gaussian')
#plt.plot(x, anal_init, 'r--', label='Fourier series reconstruction')
#plt.xlabel('x')
#plt.ylabel('u(x,0)')
#plt.title('Initial condition: Gaussian vs Fourier series')
#plt.legend()
#plt.grid(True)
#plt.show()

#max_err = np.max(np.abs(anal_init - u0))
#print(f'Max absolute error of Fourier series reconstruction at t=0: {max_err:.3e}')

# --------------------------------------------------------
# Compare numerical solutions to analytical at selected times
# --------------------------------------------------------
times_to_compare = [0, 1, 2, 5, 10, 50]

times_to_compare = [0, dt*N_iter/50, dt*N_iter/25, dt*N_iter/10, dt*N_iter/5, dt*N_iter]  # times to compare

plt.figure(figsize=(15, 8))
for idx, t_plot in enumerate(times_to_compare, 1):
    num_index = int(t_plot / dt)  # corresponding numerical step

    num_with = solution_with_jac[num_index]
    num_without = solution_without_jac[num_index]

    # Analytical solution on the same coarse grid using u0
    anal_coarse = analytical_solution_on_grid(x, t_plot, L, D, u0=u0, x_grid=x, suma=200)

    plt.subplot(2, 3, idx)
    plt.plot(x, anal_coarse, 'k-', label='Analytical')
    plt.plot(x, num_with, 'ro--', label='Numerical (with jac)')
    plt.plot(x, num_without, 'bs--', label='Numerical (without jac)')
    plt.xlim(0, L)
    plt.ylim(-1, A+1)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f't = {t_plot:.1f} s')
    plt.grid(True)
    if idx == len(times_to_compare):
        plt.legend()

plt.tight_layout()
plt.show()
