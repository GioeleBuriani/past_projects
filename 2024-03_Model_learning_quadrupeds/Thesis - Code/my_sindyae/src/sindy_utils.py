import numpy as np
from scipy.special import binom
from scipy.integrate import odeint

import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate the expected size of the library before its creation
# This is useful for defining parameter configurations and optimizations
def library_size(n, poly_order, use_goniom=False, input_type=None, include_constant=True):
    l = 0  # Initialize the library size counter

    # Adjust the number of dimensions to consider if the model order includes derivatives
    n = 2 * n  # Doubling for state variables and their derivatives

    # Calculate the total number of polynomial terms up to the specified order
    for k in range(poly_order + 1):
        l += int(binom(n + k - 1, k))  # Use binomial coefficient to calculate combinations

    # If using trigonometric functions, add sine and cosine terms for each dimension
    if use_goniom:
        l += n * 2  # Each dimension contributes sine and cosine terms

    # Include input terms based on the input type and model order
    if input_type is not None:
        l += (n / 2)  # Add one input term per two dimensions for second order model

    # Exclude the constant term from the library if not needed
    if not include_constant:
        l -= 1

    return int(l)  # Return the final calculated library size

# Function to generate a library for the SINDy algorithm for a system of order 2
def sindy_library_order2(X, dX, U, poly_order, include_goniom=False, input_type=None):
    m, n = X.shape  # Extract the dimensions of the state matrix X

    # Calculate the total number of terms in the library including polynomial and optional terms
    l = library_size(n, poly_order, include_goniom, input_type, True)

    # Initialize the library matrix with ones (for the constant term)
    library = np.ones((m, l))

    # Start filling the library from the second column
    index = 1

    # Combine the state variables and their derivatives for polynomial term generation
    X_combined = np.concatenate((X, dX), axis=1)

    # Linear terms: Add state and derivative variables directly to the library
    for i in range(2 * n):
        library[:, index] = X_combined[:, i]
        index += 1

    # Quadratic and higher order interactions: Add polynomial terms of order 2 or higher
    if poly_order > 1:
        for i in range(2 * n):
            for j in range(i, 2 * n):
                library[:, index] = X_combined[:, i] * X_combined[:, j]
                index += 1

    # Cubic interactions
    if poly_order > 2:
        for i in range(2 * n):
            for j in range(i, 2 * n):
                for k in range(j, 2 * n):
                    library[:, index] = X_combined[:, i] * X_combined[:, j] * X_combined[:, k]
                    index += 1

    # Quartic interactions
    if poly_order > 3:
        for i in range(2 * n):
            for j in range(i, 2 * n):
                for k in range(j, 2 * n):
                    for q in range(k, 2 * n):
                        library[:, index] = X_combined[:, i] * X_combined[:, j] * X_combined[:, k] * X_combined[:, q]
                        index += 1
                    
    # Quintic interactions
    if poly_order > 4:
        for i in range(2 * n):
            for j in range(i, 2 * n):
                for k in range(j, 2 * n):
                    for q in range(k, 2 * n):
                        for r in range(q, 2 * n):
                            library[:, index] = X_combined[:, i] * X_combined[:, j] * X_combined[:, k] * X_combined[:, q] * X_combined[:, r]
                            index += 1

    # Include trigonometric functions if goniometric terms are requested
    if include_goniom:
        for i in range(2 * n):
            library[:, index] = np.sin(X_combined[:, i])
            index += 1
        for i in range(2 * n):
            library[:, index] = np.cos(X_combined[:, i])
            index += 1

    # Include control input terms if specified
    if input_type is not None:
        for i in range(n):
            library[:, index] = U[:, i]
            index += 1

    return library

# Function to create a simple SINDy library for an ASLiP model
def sindy_libary_aslip(z, dz, u_latent):
    x = z[:, 0]  # Extract x component
    z = z[:, 1]  # Extract z component
    u_x = u_latent[:, 0]  # Control input in x-direction
    u_z = u_latent[:, 1]  # Control input in z-direction

    # Start the library with a column of ones (constant term)
    library = [np.ones(z.shape[0])]

    # Append state variables and inputs directly to the library list
    library.append(x)
    library.append(z)
    library.append(u_x)
    library.append(u_z)

    # Stack all the columns to form a matrix
    return np.column_stack(library)

# Function that, given the trained sindy coefficients and the library parameters, prints the learnt equations
def write_learnt_equations(sindy_coefficients, latent_dim, poly_order, include_goniom=False, input_type=None):

    # Initialize the list of equations
    equations = []

    # Iterate over the latent variables
    for state in range(latent_dim):

        # Initialize the equation string
        equation = []
        offset = 0

        # Add the state variable derivative as left side of the equation
        equation_header = 'ddz_' + str(state+1) + ' = '

        # If present, add the constant term
        if sindy_coefficients[offset, state] != 0:
            equation.append('{:.2f}'.format(sindy_coefficients[0, state]))
        offset += 1

        # Add the linear terms of the equation
        for i in range(latent_dim):
            if sindy_coefficients[offset, state] != 0:
                equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + 'z_' + str(i+1))
            offset += 1
        for i in range(latent_dim):
            if sindy_coefficients[offset, state] != 0:
                equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + 'dz_' + str(i+1))
            offset += 1

        # Add the polynomial terms of the equation
        loop_dim = 2 * latent_dim

        if poly_order > 1:
            for i in range(loop_dim):
                for j in range(i, loop_dim):
                    if i < latent_dim:
                        term1 = "z_" + str(i + 1)
                    else:
                        term1 = "dz_" + str(i - latent_dim + 1)
                    if j < latent_dim:
                        term2 = "z_" + str(j + 1)
                    else:
                        term2 = "dz_" + str(j - latent_dim + 1)
                    term = term1 + term2
                    if sindy_coefficients[offset, state] != 0:
                        equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + term)
                    offset += 1
        
        if poly_order > 2:
            for i in range(loop_dim):
                for j in range(i, loop_dim):
                    for k in range(j, loop_dim):
                        if i < latent_dim:
                            term1 = "z_" + str(i + 1)
                        else:
                            term1 = "dz_" + str(i - latent_dim + 1)
                        if j < latent_dim:
                            term2 = "z_" + str(j + 1)
                        else:
                            term2 = "dz_" + str(j - latent_dim + 1)
                        if k < latent_dim:
                            term3 = "z_" + str(k + 1)
                        else:
                            term3 = "dz_" + str(k - latent_dim + 1)
                        term = term1 + term2 + term3
                        if sindy_coefficients[offset, state] != 0:
                            equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + term)
                        offset += 1
        
        if poly_order > 3:
            for i in range(loop_dim):
                for j in range(i, loop_dim):
                    for k in range(j, loop_dim):
                        for p in range(k, loop_dim):
                            if i < latent_dim:
                                term1 = "z_" + str(i + 1)
                            else:
                                term1 = "dz_" + str(i - latent_dim + 1)
                            if j < latent_dim:
                                term2 = "z_" + str(j + 1)
                            else:
                                term2 = "dz_" + str(j - latent_dim + 1)
                            if k < latent_dim:
                                term3 = "z_" + str(k + 1)
                            else:
                                term3 = "dz_" + str(k - latent_dim + 1)
                            if p < latent_dim:
                                term4 = "z_" + str(p + 1)
                            else:
                                term4 = "dz_" + str(p - latent_dim + 1)
                            term = term1 + term2 + term3 + term4
                            if sindy_coefficients[offset, state] != 0:
                                equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + term)
                            offset += 1

        if poly_order > 4:
            for i in range(loop_dim):
                for j in range(i, loop_dim):
                    for k in range(j, loop_dim):
                        for p in range(k, loop_dim):
                            for q in range(p, loop_dim):
                                if i < latent_dim:
                                    term1 = "z_" + str(i + 1)
                                else:
                                    term1 = "dz_" + str(i - latent_dim + 1)
                                if j < latent_dim:
                                    term2 = "z_" + str(j + 1)
                                else:
                                    term2 = "dz_" + str(j - latent_dim + 1)
                                if k < latent_dim:
                                    term3 = "z_" + str(k + 1)
                                else:
                                    term3 = "dz_" + str(k - latent_dim + 1)
                                if p < latent_dim:
                                    term4 = "z_" + str(p + 1)
                                else:
                                    term4 = "dz_" + str(p - latent_dim + 1)
                                if q < latent_dim:
                                    term5 = "z_" + str(q + 1)
                                else:
                                    term5 = "dz_" + str(q - latent_dim + 1)
                                term = term1 + term2 + term3 + term4 + term5
                                if sindy_coefficients[offset, state] != 0:
                                    equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + term)
                                offset += 1

        # Add the trigonometric terms of the equation if specified
        if include_goniom:
            for i in range(latent_dim):
                if sindy_coefficients[offset, state] != 0:
                    equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + 'sin(z_' + str(i+1) + ')')
                offset += 1
            for i in range(latent_dim):
                if sindy_coefficients[offset, state] != 0:
                    equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + 'sin(dz_' + str(i+1) + ')')
                offset += 1
            for i in range(latent_dim):
                if sindy_coefficients[offset, state] != 0:
                    equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + 'cos(z_' + str(i+1) + ')')
                offset += 1
            for i in range(latent_dim):
                if sindy_coefficients[offset, state] != 0:
                    equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + 'cos(dz_' + str(i+1) + ')')
                offset += 1

        # Include the input terms
        if input_type is not None:
            for i in range(latent_dim):
                if sindy_coefficients[offset, state] != 0:
                    equation.append('{:.2f}'.format(sindy_coefficients[offset, state]) + '*' + 'u_' + str(i+1))
                offset += 1
                    
        # Combine all terms together to form a complete equation string for this latent dimension
        equation_str = ' + '.join(equation)
        equation_str = equation_header + equation_str
        equations.append(equation_str)

    # Print the learnt equations
    print('\nLearnt equations:')
    for i, equation in enumerate(equations):
        print(equation)

# Function to write the learned equations from SINDy coefficients for the ASLiP model
def write_learnt_aslip_equations(sindy_coefficients):
    basis_functions = ['z_1', 'z_2', 'u_1', 'u_2']  # Define basis functions corresponding to states and inputs

    equations = []  # List to hold the equations for each state variable

    # Loop through each state variable (2 for a 2D-SLIP model)
    for state in range(2):
        equation_terms = []  # List to collect terms of the equation

        # Setup the left-hand side of the differential equation
        equation_header = 'dd_' + ['z_1', 'z_2'][state] + ' = '

        # Include the constant term in the equation if it is non-zero
        if sindy_coefficients[0, state] != 0:
            equation_terms.append('{:.2f}'.format(sindy_coefficients[0, state]))

        # Loop through each coefficient and its corresponding basis function
        for coeff, basis_func in zip(sindy_coefficients[1:, state], basis_functions):
            if coeff != 0:  # Only include non-zero terms
                equation_terms.append('{:.2f}'.format(coeff) + '*' + basis_func)

        # Combine all terms to form the full equation string
        equation_str = ' + '.join(equation_terms)
        equation_str = equation_header + equation_str
        equations.append(equation_str)

    # Output the learned equations
    print('\nLearnt equations:')
    for equation in equations:
        print(equation)

# Function to perform batch simulations of a system using SINDy identified dynamics
def sindy_batch_simulate_order2(t, real_z, real_dz, u, sindy_coeff, lib_type, poly_order, include_goniom, input_type=None, batch_size=5):
    # Initialize simulation arrays for state (x), first derivative (dx), and second derivative (ddx)
    sim_x = np.empty_like(real_z)
    sim_dx = np.empty_like(real_dz)
    sim_ddx = np.empty_like(real_z)

    batch_t = t[:batch_size]  # Define the time vector for the first batch

    # Loop over all batches to simulate the dynamics
    for i in range(real_z.shape[0] // batch_size):
        # Prepare initial conditions for the current batch
        batch_initial_state = np.concatenate((real_z[i*batch_size], real_dz[i*batch_size]))
        batch_u = u[i*batch_size : (i+1)*batch_size]  # Select the control inputs for the batch

        # Define a function to interpolate control inputs during integration
        u_func = lambda t, u=batch_u, batch_t=batch_t: u[np.searchsorted(batch_t, t) % len(batch_u)]

        # Simulate dynamics using the ODE solver
        batch_sol = odeint(learned_system_dynamics, batch_initial_state, batch_t, args=(u_func, sindy_coeff, lib_type, poly_order, include_goniom, input_type))

        # Extract simulated state (x) and first derivative (dx) from the solution
        batch_sim_x = batch_sol[:, :len(real_z[0])]
        batch_sim_dx = batch_sol[:, len(real_z[0]):]
        batch_sim_ddx = np.zeros_like(batch_sim_x)  # Initialize second derivative array

        # Calculate the second derivative for each timestep in the batch
        for j in range(batch_t.size):
            batch_sim_ddx[j] = learned_system_dynamics(batch_sol[j, :], batch_t[j], u_func, sindy_coeff, lib_type, poly_order, include_goniom, input_type)[len(real_z[0]):]

        # Store the simulated results back into the main arrays
        sim_x[i * batch_size : (i + 1) * batch_size] = batch_sim_x
        sim_dx[i * batch_size : (i + 1) * batch_size] = batch_sim_dx
        sim_ddx[i * batch_size : (i + 1) * batch_size] = batch_sim_ddx

    return sim_x, sim_dx, sim_ddx  # Return the simulated data arrays

# Function to perform hybrid batch simulation of a dynamic system using two sets of SINDy coefficients
def hybrid_sindy_batch_simulate_order2(t, real_z, real_dz, u, flight_indices, sindy_coeff_1, sindy_coeff_2, lib_type, poly_order, include_goniom, input_type=None, batch_size=5):
    sim_z = np.empty_like(real_z)  # Array to hold the simulated states z
    sim_dz = np.empty_like(real_z)  # Array to hold the simulated first derivatives dz
    sim_ddz = np.empty_like(real_z)  # Array to hold the simulated second derivatives ddz

    # Process the data in batches
    for i in range(real_z.shape[0] // batch_size):
        # Determine which indices within the current batch correspond to flight phases
        batch_flight_indices = set(idx - i*batch_size for idx in flight_indices if i*batch_size <= idx < (i+1)*batch_size)
        # Concatenate current state z and dz for initial conditions
        batch_initial_state = np.concatenate((real_z[i*batch_size], real_dz[i*batch_size]))

        # Identify transitions within the batch between different dynamic modes
        sub_batches = []
        current_sub_batch = []
        for j in range(batch_size):
            current_sub_batch.append(j)
            in_flight = j in batch_flight_indices
            next_in_flight = (j + 1) in batch_flight_indices if j + 1 < batch_size else not in_flight
            if in_flight != next_in_flight or j == batch_size - 1:
                sub_batches.append((current_sub_batch, in_flight))
                current_sub_batch = []

        sub_batch_initial_state = batch_initial_state.copy()
        # Simulate each sub-batch using appropriate SINDy coefficients
        for sub_batch, in_flight in sub_batches:
            sub_batch_t = t[i*batch_size + sub_batch[0] : i*batch_size + sub_batch[-1] + 1]
            sub_batch_u = u[i*batch_size + sub_batch[0] : i*batch_size + sub_batch[-1] + 1]
            # Define a control function for the current sub-batch
            u_func = lambda t, batch_u=sub_batch_u, batch_t=sub_batch_t: batch_u[np.searchsorted(batch_t, t) % len(batch_u)]
            # Select the appropriate SINDy coefficients based on flight status
            sindy_coeff = sindy_coeff_2 if in_flight else sindy_coeff_1
            # Solve the system dynamics over the sub-batch interval
            sol = odeint(learned_system_dynamics, sub_batch_initial_state, sub_batch_t, args=(u_func, sindy_coeff, lib_type, poly_order, include_goniom, input_type))
            sub_batch_initial_state = sol[-1, :]

            # Store simulation results and compute second derivatives
            for k, index in enumerate(sub_batch):
                sim_z[i * batch_size + index] = sol[k, :len(real_z[0])]
                sim_dz[i * batch_size + index] = sol[k, len(real_z[0]):]
                # Calculate second derivatives using system dynamics at each time step
                current_state = sol[k, :]
                current_time = sub_batch_t[k]
                ddz = learned_system_dynamics(current_state, current_time, u_func, sindy_coeff, lib_type, poly_order, include_goniom, input_type)[len(real_z[0]):]
                sim_ddz[i * batch_size + index] = ddz

    return sim_z, sim_dz, sim_ddz  # Return arrays containing simulated system states and derivatives

# Function to calculate the system dynamics based on the current state and control inputs
def learned_system_dynamics(state, t, u_func, sindy_coef, lib_type, poly_order, include_goniom, input_type=None):
    n = len(state) // 2  # Half the length of state to separate position (x) and velocity (dx)

    x = state[:n].reshape((1, n))  # Current positions
    dx = state[n:].reshape((1, n))  # Current velocities

    u_current = u_func(t).reshape(1, -1)  # Get current control inputs from the function

    # Generate the appropriate SINDy library based on the specified library type
    if lib_type == 'general':
        sindy_lib = sindy_library_order2(x, dx, u_current, poly_order, include_goniom, input_type)
    elif lib_type == 'aslip':
        sindy_lib = sindy_libary_aslip(x, dx, u_current)

    ddx = np.dot(sindy_lib, sindy_coef)  # Compute acceleration (ddx) from the SINDy library and coefficients

    return np.concatenate((dx[0], ddx[0]))  # Return the concatenated derivatives for integration

# Main function
def main():

    # Create a dummy sindy coefficients matrix for testing the aslip function
    sindy_coefficients = np.ones((5, 2))

    # Set a random number of random coefficients to zero
    n = np.random.randint(0, 10)
    for i in range(n):
        sindy_coefficients[np.random.randint(0, 5), np.random.randint(0, 2)] = 0

    write_learnt_aslip_equations(sindy_coefficients)

# Execute the main function
if __name__ == "__main__":
    main()