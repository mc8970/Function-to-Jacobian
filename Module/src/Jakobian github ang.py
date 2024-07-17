#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sym
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

def jacobian(bulk, var_and_lengths_dict, left_boundary, right_boundary, lr_cut, values_dict):
    """
    Analytically constructs the Jacobian matrix for the given system of equations.

    Parameters:
    - bulk: a function that takes indices and returns a sympy expression for equations in a multivariable system.
    - var_and_lengths_dict: a dictionary specifying the number of components for each variable.
    - left_boundary: a function that returns the expression for the left boundary condition.
    - right_boundary: a function that returns the expression for the right boundary condition.
    - lr_cut: a dictionary where the first element is a list of the left boundary condition lengths for each variable,
      and the second is a list of the right boundary condition lengths for each variable.
    - values_dict: a dictionary containing the values to substitute into the Jacobian matrix.

    Returns:
    - jac: A list containing the analytical Jacobian matrix and the Jacobian matrix with substituted values.
    """
    # Generate symbols for all variables in the variable vectors
    variable_vectors = {var: [sym.symbols(f'{var}{i}') for i in range(var_and_lengths_dict[var])] 
                        for var in var_and_lengths_dict}
    
    # Flatten the list of all variable vectors with their components
    all_components = [component for components in variable_vectors.values() for component in components]
    
    all_derivatives = []

    # Adding the left boundary condition derivatives
    if left_boundary:
        left_derivatives = left_boundary(variable_vectors)
        for derivative in left_derivatives:
            all_derivatives.append([sym.diff(derivative, component) for component in all_components])

    # Adding bulk function derivatives, taking into account the lr_cut
    for var, components in variable_vectors.items():
        left_cut = lr_cut[0][var]
        right_cut = lr_cut[1][var]
        
        for i in range(left_cut, len(components) - right_cut):
            func = bulk(var, i, variable_vectors)
            
            func_derivatives = []
            
            for component in all_components:
                derivative = sym.diff(func, component)
                func_derivatives.append(derivative)
            
            all_derivatives.append(func_derivatives)

    # Adding the right boundary condition derivatives
    if right_boundary:
        right_derivatives = right_boundary(variable_vectors)
        for derivative in right_derivatives:
            all_derivatives.append([sym.diff(derivative, component) for component in all_components])

    # Creating a Jacobian matrix based on the derivatives. This is unsorted
    unsorted_jacobian = sym.Matrix(all_derivatives)
    
    # Create a list of individual variable lengths
    variable_lengths = list(var_and_lengths_dict.values())
    
    def move_down(matrix, variable_lengths):
        """
        Correctly arranges the upper rows (boundary conditions).

        Parameters:
        - matrix: the unsorted Jacobian matrix.
        - variable_lengths: list of lengths of variables.

        Returns:
        - Matrix with the upper rows properly arranged.
        """
        shift_values = []
        for i, num in enumerate(variable_lengths):
            if i == 0:
                shift_values.append(0)
            else:
                shift_values.append((len(variable_lengths) - i - 1) + sum(variable_lengths[:i]) - i - 1)

        shift_values.pop(0)

        for positions in shift_values:
            if len(matrix) < 2:
                return matrix

            row_to_move = matrix.row(1)
            matrix.row_del(1)
            new_index = min(1 + positions, len(matrix) - 1)
            matrix = matrix.row_insert(new_index, row_to_move)

        return matrix
    
    # Creating a Jacobian matrix with sorted upper rows
    upper_arranged_jacobian = sym.Matrix(move_down(unsorted_jacobian, variable_lengths))
    
    def move_up(matrix, variable_lengths):
        """
        Correctly arranges the lower rows (boundary conditions).

        Parameters:
        - matrix: the upper-arranged Jacobian matrix.
        - variable_lengths: list of lengths of variables.

        Returns:
        - Matrix with the lower rows properly arranged.
        """
        variable_lengths = variable_lengths[::-1]
        shift_values = []
        for i, num in enumerate(variable_lengths):
            if i == 0:
                shift_values.append(0)
            else:
                shift_values.append((len(variable_lengths) - i - 1) + sum(variable_lengths[:i]) - 1)

        shift_values.pop(0)
        
        for positions in shift_values:
            if len(matrix) < 2:
                return matrix

            row_to_move = matrix.row(-2)
            matrix.row_del(-2)
            new_index = min(-1 - positions, len(matrix))
            matrix = matrix.row_insert(new_index, row_to_move)

        return matrix

    # Creating the final analytically calculated Jacobian matrix
    jacobian = sym.Matrix(move_up(upper_arranged_jacobian, variable_lengths))
    
    def insert_values(values_dict, matrix):
        """
        Inserts values into the matrix.

        Parameters:
        - values_dict: dictionary containing values to substitute into the matrix.
        - matrix: the Jacobian matrix with symbolic variables.

        Returns:
        - Numpy array of the Jacobian matrix with substituted values.
        """
        symbols = list(matrix.free_symbols)
        lambdified_func = sym.lambdify(symbols, matrix, 'numpy')
        matrix_np = lambdified_func(**values_dict)
        return matrix_np
    
    # Substituting the values into the Jacobian matrix
    jacobian_values = insert_values(values_dict, jacobian)
    
    # The output of our function, a list with the analytical and value-substituted Jacobian matrix
    jac = [jacobian, jacobian_values]
    
    def print_red(text):
        """
        Prints text in red

        Parameters:
        - text: the text to be printed in red.
        """
        print("\033[31m" + text + "\033[0m")  # 31 is the ANSI color code for red

    # Error notification if the Jacobian is incorrectly formed
    if not np.all(np.diagonal(jacobian_values)):
        print_red("WARNING! Zeros on the diagonal.")
    if jacobian.shape[0] != jacobian.shape[1]:
        print_red("WARNING! Jacobian is not a square matrix.")
    
    return jac

# Defining the bulk and boundary functions
def bulk(var, i, vars):
    u = vars['u']
    D, dt, dx, u_old = sym.symbols('D dt dx u_old')
    if var == 'u':
        return D * (u[i - 1] - 2 * u[i] + u[i + 1]) * dt / dx**2 + u_old - u[i]

def left_boundary(vars):
    u0 = vars['u'][0]
    u1 = vars['u'][1]
    dx = sym.symbols('dx')
    return [(u1 - u0) / dx**2]

def right_boundary(vars):
    u_n = vars['u'][-1]
    u_nn = vars['u'][-2]
    dx = sym.symbols('dx')
    return [(-u_n + u_nn) / dx**2]

# Number of primary variables, left, right cut values and other variables values
var_and_lengths_dict = {'u': 5}
lr_cut = [{'u': 1}, {'u': 1}]
values_dict = {'D': 0.5, 'dt': 0.5, 'dx': 10/4}

# Generating the analytical Jacobian matrix of our problem
jacobian = jacobian(bulk, var_and_lengths_dict, left_boundary, right_boundary, lr_cut, values_dict)

# Function outputs
jacobian[0]
# jacobian[1]


# In[2]:


# Preparing the function for the root solver
def fun(u):
    u_xx = np.zeros(Nx)
    u_xx[0] = (u[1] - u[0]) / dx**2 # Left boundary condition
    u_xx[-1] = (- u[-1] + u[-2]) / dx**2 # Right boundary condition
    for i in range(1, Nx - 1): # Diferential scheme for the points in the middle (bulk function)
        u_xx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2 
    return u - u0 - dt * D * u_xx # There is a 0 on the other side of the equation


# In[3]:


import sympy as sym
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import timeit


L = 10 # Lenght
D = 0.5 # Diffusion coefficient
T = 100 # Whole time
Nx = 5 # Discretisation of space
dt = 0.5 # Time step

dx = L / (Nx - 1) # Space step
x = np.linspace(0, L, Nx, endpoint=True) # Space points x values

A = 10.0  # Max particle number
mu = 5  # Center of distrobution
sigma = 0.25  # Standard deviation
initial_u0 = A * np.exp(-0.5 * ((x - mu) / sigma)**2) # To reset when the other function is called
u0 = A * np.exp(-0.5 * ((x - mu) / sigma)**2)

def jac(u):
    return jacobian[1]


# In[4]:


import timeit
def solution_root(our_jac, u0):
    t = 0
    solution = [u0.copy()]  # Saving the results
    times = [t]  # Saving the times

    while t < T:
        if our_jac:
            sol = root(fun, u0, jac=jac)
        else:
            sol = root(fun, u0)
        u0 = sol.x
        t += dt  # New iteration
        solution.append(u0.copy())  # Save the solution of an iteration
        times.append(t)  # Save the time of the iteration
    
    return solution, times


# In[5]:


time_without_jac = timeit.timeit(lambda: solution_root(our_jac=False, u0=initial_u0.copy()), number=2)
time_with_jac = timeit.timeit(lambda: solution_root(our_jac=jac, u0=initial_u0.copy()), number=2)
print(f"Time without Jacobian matrix argument: {time_without_jac:.4f} [s]")
print(f"Time with Jacobian Matrix argument: {time_with_jac:.4f} [s]")
print(f"The differance is {(time_with_jac - time_without_jac):.4f} [s] or {((time_with_jac - time_without_jac)/time_without_jac)*100:.4f}% faster/slower (- faster with jac, + slower with jac)")


# In[6]:


x1 = [0,5, 10, 50, 100]
ybz = [0,0.0373, 0.0714, 0.7377, 2.3704]
yz = [0,0.024, 0.03, 0.2882, 0.9147]
yproc = [0,27.59, 58.5, 60.9, 61.4]


#plt.plot(x1, ybz, label=f'Time without Jacobian', linewidth=2, color='red')
#plt.plot(x1, yz, label=f'Time with Jacobian', linewidth=2, color='black')
#plt.xlabel(f'Number of points')
#plt.ylabel(f'Time for execution [s]')
#plt.fill_between(x1, ybz, yz, color='green', interpolate=True, label=f'Time saved')
#plt.legend()
#plt.grid(True)

plt.plot(x1, yproc)
plt.ylim(0, 100)
plt.xlabel(f'Number of points')
plt.ylabel(f'Decrease in execution time [%]')
plt.grid(True)

plt.show()


# In[7]:


solution_without_jac, _ = solution_root(False, initial_u0.copy())
solution_with_jac, _ = solution_root(True, initial_u0.copy())

# Comparing the results
differance = np.linalg.norm(solution_without_jac[-1] - solution_with_jac[-1])
print(f"The difference in final results: {differance:.4e}")

# Calculating the entire differance
entire_differance = 0
for i, j in zip(solution_without_jac, solution_with_jac):
    entire_differance += np.linalg.norm(abs(i - j))

print(f"The entire difference: {entire_differance:.4e}")


# ANALYTICAL SOLUTION

# In[8]:


import numpy as np
from scipy.integrate import simpson as simps
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 10
D = 0.5
T = 100
points = 100000+1
dt = 1
anal_dx = L / (points - 1)
anal_x = np.linspace(0, L, points, endpoint=True)

A = 10.0
mu = L / 2
sigma = 0.25

# Initial condition
anal_init_u0 = A * np.exp(-0.5 * ((anal_x - mu) / sigma)**2)

# Defining the analytical solution depending on the number of series components(suma argument)
def analytical_solution(x, t, L, D, suma, points):
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


# Initial values
limit = 10**-12
number_of_components = 95
difference = np.inf

# While loop for seeking the number components, where the differance from the initial condition is less than the 12th decimal point.
while difference > limit:
    solution = analytical_solution(anal_x, 0, L, D, number_of_components, points)
    difference = np.max(np.abs(solution - anal_init_u0))
    
    # Real-Time writting of iteration
    print(f"Iteration: {number_of_components}")
    print(f"Difference: {difference:.12f}")
    
    number_of_components += 1

# The output ends
print(f"Converges with the following number of components : {number_of_components}")

solution = analytical_solution(anal_x, 0.1, L, D, 95, points)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the initial condition
axs[0].plot(anal_x, anal_init_u0, label='Gaussian initial condition')
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('u(x,t) [Number of Particles]')
axs[0].set_title('Initial condition')
axs[0].legend(loc='upper right')
axs[0].grid(True)

# Plotting
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

# Gathering solutions
specific_solutions = {}
for x in x_val:
    index = np.where(np.isclose(anal_x, x))[0][0]
    specific_solutions[x] = solution[index]

# Outputting values
for x, val in specific_solutions.items():
    print(f"Value at x = {x}: {val}")

array_solutions = []
for x, val in specific_solutions.items():
    values = val
    array_solutions.append(val)
    
array_solutions = np.array(array_solutions)


# In[9]:


time = 10
dt = 0.5
index = int(time / dt)


solution = analytical_solution(anal_x, time, L, D, number_of_components, points)
# Values at specific points
x_val = [0, 2.5, 5, 7.5, 10]

# Gathering results
#specific_solutions = {}
#for x in x_val:
 #   index = np.where(np.isclose(anal_x, x))[0][0]
  #  specific_solutions[x] = solution[index]

#array_solutions = []
#for x, val in specific_solutions.items():
 #   value = val
  #  array_solutions.append(val)
    
#array_solutions = np.array(array_solutions)


with_jac = 0
without_jac = 0
t = np.linspace(0, time, 10, endpoint=True)
# Differences
for t_i in t:
    t_index = int(t_i/dt)
    anal_solution = analytical_solution(anal_x, t_i, L, D, number_of_components, points)
    specific_solutions = {}
    for x in x_val:
        ind = np.where(np.isclose(anal_x, x))[0][0]
        specific_solutions[x] = anal_solution[ind]

    solutions = []
    for x, val in specific_solutions.items():
        values = val
        solutions.append(val)
        array_solutions = np.array(array_solutions)
    for i, j, k in zip(solution_with_jac[t_index], solution_without_jac[t_index], array_solutions):
        with_jac += np.linalg.norm(abs(i - k))
        without_jac += np.linalg.norm(abs(j - k))

print(f'Jac {with_jac}')
print(f'Without jac {without_jac}')
print(f'Difference {with_jac-without_jac} + bigger with jac, - bigger without jac oz. {100*((with_jac-without_jac)/without_jac)}%')

print(f'With jac values: {solution_with_jac[index]}')
print(f'Without jac values: {solution_without_jac[index]}')
print(f'Analytical solutions at t = {time} are: {array_solutions}')

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

