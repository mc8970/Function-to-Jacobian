#!/usr/bin/env python
# coding: utf-8

import sympy as sym
import numpy as np

def analytical_jacobian(bulk, var_and_length_dict, left_boundary, right_boundary, lr_cut, value_dict):
    """
    Differentiates functions given in functional form to generate the Jacobian.

    Parameters:
    - bulk: function that takes indices and returns a sympy expression for equations 
            in a multivariable system
    - var_and_length_dict: dictionary that defines the number of components for each variable
    - left_boundary: function that returns an expression for the left boundary condition
    - right_boundary: function that returns an expression for the right boundary condition
    - lr_cut: dictionary where the first element is a list of left boundary condition lengths 
              for each variable, and the second element is a list of right boundary condition lengths 
              for each variable
    - value_dict: dictionary mapping symbols to numerical values

    Returns:
    - Jacobian matrix for the list of equations, including boundary conditions
    """
    # Generate symbols for all variables in variable vectors
    variable_vectors = {
        var: [sym.symbols(f'{var}{i}') for i in range(var_and_length_dict[var])] 
        for var in var_and_length_dict
    }
    
    # Flatten the list of all variable vectors with their components
    all_components = [component for components in variable_vectors.values() for component in components]
    
    all_derivatives = []

    # Add left boundary condition
    if left_boundary:
        left_derivatives = left_boundary(variable_vectors)  # Multiple boundary conditions possible
        for derivative in left_derivatives:
            all_derivatives.append([sym.diff(derivative, component) for component in all_components])

    # Process each variable independently
    for var, components in variable_vectors.items():
        left_cut = lr_cut[0][var]
        right_cut = lr_cut[1][var]
        
        # Generate expression for each index, considering left and right boundaries
        for i in range(left_cut, len(components) - right_cut):
            fun = bulk(var, i, variable_vectors)
            
            fun_derivatives = []
            
            # Differentiate the generated function with respect to each component
            for component in all_components:
                derivative = sym.diff(fun, component)
                fun_derivatives.append(derivative)
            
            all_derivatives.append(fun_derivatives)

    # Add right boundary condition
    if right_boundary:
        right_derivatives = right_boundary(variable_vectors)  # Multiple boundary conditions possible
        for derivative in right_derivatives:
            all_derivatives.append([sym.diff(derivative, component) for component in all_components])

    # Create matrix based on derivatives. This is unsorted
    unsorted_jacobian = sym.Matrix(all_derivatives)
    
    # Create a list of individual variable lengths
    variable_lengths = list(var_and_length_dict.values())
    
    # Function that correctly sorts the upper rows (BCs)
    def move_down(matrix, variable_lengths):
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
    
    upper_sorted_jacobian = sym.Matrix(move_down(unsorted_jacobian, variable_lengths))
    
    # Function that correctly sorts the lower rows (BCs)
    def move_up(matrix, variable_lengths):
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

    sorted_jacobian = sym.Matrix(move_up(upper_sorted_jacobian, variable_lengths))
    
    # Function that inserts values into the matrix
    def insert_values(value_dict, matrix):
        symbols = list(matrix.free_symbols)
        lambdified_func = sym.lambdify(symbols, matrix, 'numpy')
        matrix_np = lambdified_func(**value_dict)
        return matrix_np
    
    jacobian_values = insert_values(value_dict, sorted_jacobian)
    jac = [sorted_jacobian, jacobian_values]
    
    # Error reporting if something is wrong with the Jacobian
    if not np.all(np.diagonal(jacobian_values)):
       print(f"WARNING! Zeros on the diagonal.")
    if sorted_jacobian.shape[0] != sorted_jacobian.shape[1]:
       print(f"WARNING! Jacobian is not a square matrix.")
    return jac