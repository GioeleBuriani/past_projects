import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib as mpl

# Function to simulate hybrid Active Spring Loaded Inverted Pendulum (ASLIP) dynamics in batches
def hybrid_aslip_batch_simulate(t, real_pos, real_vel, u, flight_indices, batch_size=5):
    # Set constants for simulation: gravity, mass, spring constant, and leg rest length
    g = -9.81  # acceleration due to gravity (m/s^2)
    m = 12.01  # mass of the ASLIP model (kg)
    k_s = 250  # spring constant (N/m)
    l0 = 0.32  # natural length of the leg (m)
    d = real_pos[-1, 0]  # initial horizontal displacement
    le = real_pos[-1, 1]  # initial leg extension

    # Initialize arrays to store simulated positions, velocities, and accelerations
    sim_pos = np.empty_like(real_pos)
    sim_vel = np.empty_like(real_pos)
    sim_acc = np.empty_like(real_pos)

    # Process the data in batches
    for i in range(real_pos.shape[0] // batch_size):
        # Calculate indices for the current batch that are in flight
        batch_flight_indices = set(idx - i*batch_size for idx in flight_indices if i*batch_size <= idx < (i+1)*batch_size)
        batch_initial_state = np.concatenate((real_pos[i*batch_size], real_vel[i*batch_size]))

        # Segment the batch into sub-batches based on changes between flight and contact phases
        sub_batches = []
        current_sub_batch = []
        for j in range(batch_size):
            current_sub_batch.append(j)
            in_flight = j in batch_flight_indices
            next_in_flight = (j + 1) in batch_flight_indices if j + 1 < batch_size else not in_flight
            if in_flight != next_in_flight or j == batch_size - 1:
                sub_batches.append((current_sub_batch, in_flight))
                current_sub_batch = []

        # Initialize state variables for sub-batches
        sub_batch_initial_state = batch_initial_state.copy()
        # Process each sub-batch separately
        for sub_batch, in_flight in sub_batches:
            sub_batch_t = t[i*batch_size + sub_batch[0] : i*batch_size + sub_batch[-1] + 1]
            sub_batch_u = u[i*batch_size + sub_batch[0] : i*batch_size + sub_batch[-1] + 1]
            u_func = lambda t, batch_u=sub_batch_u, batch_t=sub_batch_t: batch_u[np.searchsorted(batch_t, t) % len(batch_u)]

            # Choose the dynamics function based on whether the model is in flight or in contact
            if in_flight:
                sol = odeint(aslip_dynamics_flight, sub_batch_initial_state, sub_batch_t, args=(g,))
            else:
                sol = odeint(aslip_dynamics_contact, sub_batch_initial_state, sub_batch_t, args=(u_func, g, m, k_s, l0, d, le))
            sub_batch_initial_state = sol[-1, :]

            # Store the results for position, velocity, and calculate acceleration
            for k, index in enumerate(sub_batch):
                sim_pos[i * batch_size + index] = sol[k, :len(real_pos[0])]
                sim_vel[i * batch_size + index] = sol[k, len(real_pos[0]):]
                current_state = sol[k, :]
                current_time = sub_batch_t[k]
                # Compute acceleration based on current state and dynamics
                if in_flight:
                    sim_acc[i * batch_size + index] = aslip_dynamics_flight(current_state, current_time, g)[len(real_pos[0]):]
                else:
                    sim_acc[i * batch_size + index] = aslip_dynamics_contact(current_state, current_time, u_func, g, m, k_s, l0, d, le)[len(real_pos[0]):]

    return sim_pos, sim_vel, sim_acc
    
# Function to model the contact dynamics of an Active Spring Loaded Inverted Pendulum (ASLIP)
def aslip_dynamics_contact(state, t, u_func, g, m, k_s, l0, d, le):
    # Extract the position and velocity from the state vector
    pos = state[:2].reshape((1, 2))
    vel = state[2:].reshape((1, 2))

    # Compute control forces at time t using the provided control function
    u = u_func(t)
    u_x = u[0]
    u_z = u[1]

    # Assign position and velocity components to variables for clarity
    x = pos[0, 0]
    z = pos[0, 1]
    dx = vel[0, 0]
    dz = vel[0, 1]

    # Prevent penetration into the ground by setting the vertical position to zero if negative
    if z < 0:
        z = 0

    # Modify x and l0 when time passes a specific threshold, representing a phase change or control action
    if t > 0.5:
        x = d - x
        l0 = le

    # Calculate the current leg length from the origin
    l = np.sqrt(x**2 + z**2)

    # Compute spring forces based on leg compression
    F_kx = np.abs(k_s * (l0 - l)) * (x / l)
    F_kz = k_s * (l0 - l) * (z / l)

    # Compute accelerations using Newton's second law and add control inputs
    ddx = F_kx / m + u_x / m
    ddz = F_kz / m + g + u_z / m

    # Return the derivatives of state as a flattened array
    return np.concatenate([vel, np.array([[ddx, ddz]])], axis=1).reshape(-1)

# Function to model the flight dynamics of an Active Spring Loaded Inverted Pendulum (ASLIP)
def aslip_dynamics_flight(state, t, g):
    # Extract the position and velocity from the state vector
    pos = state[:2].reshape((1, 2))
    vel = state[2:].reshape((1, 2))

    # Only gravitational acceleration affects the pendulum in flight
    acc = np.array([[0, g]])

    # Return the derivatives of state as a flattened array, indicating no horizontal acceleration
    return np.concatenate([vel, acc], axis=1).reshape(-1)

# Function to plot comparison of real and simulated ASLIP x and z positions
def plot_aslip(t, real_data, simulated_data):
    # Create a figure with two subplots stacked vertically
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot real and simulated x positions
    axs[0].plot(t, real_data[:, 0], label='Real x')
    axs[0].plot(t, simulated_data[:, 0], label='Simulated x')
    axs[0].set_title('X position')  # Title for the x position plot

    # Plot real and simulated z positions
    axs[1].plot(t, real_data[:, 1], label='Real z')
    axs[1].plot(t, simulated_data[:, 1], label='Simulated z')
    axs[1].set_title('Z position')  # Title for the z position plot

    # Add legends to both subplots
    for ax in axs:
        ax.legend()

    # Display the plot
    plt.show(block=False)

# Function to plot real, ASLIP, and learned body data for x and z positions
def plot_aslip_3(time, body_data, aslip_body_data=None, learned_body_data=None, save_filename=None):
    # Set font properties for the plot using Matplotlib
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20  # Increase base font size for better readability
    mpl.rcParams['axes.titlesize'] = 25  # Increase font size for subplot titles
    mpl.rcParams['axes.labelsize'] = 25  # Increase font size for axis labels
    mpl.rcParams['xtick.labelsize'] = 15  # Font size for x-axis tick labels
    mpl.rcParams['ytick.labelsize'] = 15  # Font size for y-axis tick labels
    mpl.rcParams['legend.fontsize'] = 20  # Font size for legends
    mpl.rcParams['lines.linewidth'] = 3   # Increase line width for better visibility

    # Scale learned body data if present
    if learned_body_data is not None:
        learned_body_data = learned_body_data * 10

    # Determine the minimum and maximum values from the body data to set y-axis limits
    global_min = np.min(body_data)
    global_max = np.max(body_data)

    # Update global minimum and maximum based on ASLIP and learned data if available
    if aslip_body_data is not None:
        global_min = min(global_min, np.min(aslip_body_data))
        global_max = max(global_max, np.max(aslip_body_data))
    if learned_body_data is not None:
        global_min = min(global_min, np.min(learned_body_data))
        global_max = max(global_max, np.max(learned_body_data))

    # Calculate margin for y-axis limits to ensure data does not touch the axes
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 5.4))
    axs = axs.flatten()  # Flatten the axis array for easier indexing

    # Define plot names for x and z positions
    plot_names = ['Body linear Position x', 'Body linear Position z']

    # Plot the real, ASLIP, and learned data for x and z positions
    for i in range(2):
        axs[i].plot(time, body_data[:, i], label='Real data')
        if aslip_body_data is not None:
            axs[i].plot(time, aslip_body_data[:, i], label='Aslip data', color='purple')
        if learned_body_data is not None:
            axs[i].plot(time, learned_body_data[:, i], label='Learned data', color='orange')

        # Set titles and labels for each subplot
        axs[i].set_title(plot_names[i])
        axs[i].set_xlabel('Time (s)')
        if i == 0:  # Set y-axis label for the first subplot only
            axs[i].set_ylabel('Position (m)')

        # Set y-axis limits for consistency across plots
        axs[i].set_ylim(y_min, y_max)
        axs[i].grid(True)  # Enable grid for better data visualization
        axs[i].legend()

    # Adjust layout to prevent overlap of plot elements
    plt.tight_layout()

    # Save the plot to a file if a filename is provided, otherwise display the plot
    if save_filename:
        plt.savefig(save_filename, dpi=100)  # Save the figure with high resolution
        plt.show(block=False)
    else:
        plt.show(block=False)  # Display the plot without blocking further code execution
