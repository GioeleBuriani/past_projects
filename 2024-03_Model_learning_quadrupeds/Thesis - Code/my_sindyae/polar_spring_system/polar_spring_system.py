import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to generate data for the polar spring system
def get_pss_data(n_ics, t, coord='polar', real=False, noise_strength=0):
    # Determine the number of time steps and the dimensionality of input data
    n_steps = t.size
    input_dim = 2

    # Generate random initial conditions for theta, r, theta_dot, and r_dot
    theta0 = np.random.uniform(-np.pi / 2, np.pi / 2, n_ics)
    r0 = np.random.uniform(0.5, 2.5, n_ics)
    theta_dot0 = np.random.uniform(-0.5, 0.5, n_ics)
    r_dot0 = np.random.uniform(-0.5, 0.5, n_ics)
    ics = np.column_stack((theta0, r0, theta_dot0, r_dot0))

    # Generate the data set
    data = generate_pss_data(ics, t, normalization=np.array([1/10, 1/10]), coord=coord, real=real)

    # Add noise to the data
    data['x'] = data['x'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['dx'] = data['dx'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['ddx'] = data['ddx'].reshape((-1, input_dim)) + noise_strength * np.random.randn(n_steps * n_ics, input_dim)

    return data

# Function to generate polar spring system data
def generate_pss_data(ics, t, normalization=None, coord='polar', real=False, k_arm=30.0, b_arm=0.5, k_angle=30.0, b_angle=0.5, m=1.0, arm_length=1.5):
    # Initialize variables
    n_ics = ics.shape[0]
    n_steps = t.size
    d = 2

    # Initialize arrays for position, velocity, and acceleration
    x = np.zeros((n_ics, n_steps, d))
    dx = np.zeros(x.shape)
    ddx = np.zeros(x.shape)

    # Simulate the polar spring system for each set of initial conditions
    for i in range(n_ics):
        x[i], dx[i], ddx[i] = simulate_pss(ics[i], t, k_arm=k_arm, b_arm=b_arm, k_angle=k_angle, b_angle=b_angle, m=m, arm_length=arm_length)

    # Apply differentiation and normalization if required
    if real:
        dx = np.gradient(x, t, axis=1)
        ddx = np.gradient(dx, t, axis=1)
    if normalization is not None:
        x *= normalization
        dx *= normalization
        ddx *= normalization

    # Convert from polar to cartesian coordinates if specified
    if coord == 'cartesian':
        x, dx, ddx = polar_to_cartesian(x, dx, ddx)

    # Create a dictionary to store the data
    data = {"t": t, "x": x, "dx": dx, "ddx": ddx}

    return data

# Function to simulate the polar spring system
def simulate_pss(x0, t, k_arm=30.0, b_arm=0.5, k_angle=5000.0, b_angle=50.0, m=1.0, arm_length=1.5):
    # Define the gravitational acceleration
    g = 9.81

    # Define the differential equation of the system
    def f(y, t):
        theta, r, theta_dot, r_dot = y
        theta_ddot = g * r * np.sin(theta) - k_angle * theta / m - b_angle * theta_dot / m
        r_ddot = - g * np.cos(theta) - k_arm * (r - arm_length) / m - b_arm * r_dot / m
        return [theta_dot, r_dot, theta_ddot, r_ddot]

    # Solve the differential equation
    sol = odeint(f, x0, t)
    x = sol[:, 0:2]
    dx = sol[:, 2:4]

    # Calculate the second derivative (acceleration)
    ddx = np.zeros_like(x)
    for i in range(t.size):
        ddx[i] = f(sol[i], t[i])[2:4]

    return x, dx, ddx

# Function to convert polar coordinates to cartesian coordinates
def polar_to_cartesian(x, dx, ddx):
    # Extract polar coordinates and their derivatives
    theta = x[:, :, 0]
    r = x[:, :, 1]
    theta_dot = dx[:, :, 0]
    r_dot = dx[:, :, 1]
    theta_ddot = ddx[:, :, 0]
    r_ddot = ddx[:, :, 1]

    # Convert to cartesian coordinates
    x_cart = r * np.sin(theta)
    y_cart = r * np.cos(theta)
    x_dot = r_dot * np.sin(theta) + r * theta_dot * np.cos(theta)
    y_dot = r_dot * np.cos(theta) - r * theta_dot * np.sin(theta)
    x_ddot = r_ddot * np.sin(theta) + 2 * r_dot * theta_dot * np.cos(theta) + r * theta_ddot * np.cos(theta) - r * theta_dot**2 * np.sin(theta)
    y_ddot = r_ddot * np.cos(theta) - 2 * r_dot * theta_dot * np.sin(theta) - r * theta_ddot * np.sin(theta) - r * theta_dot**2 * np.cos(theta)

    # Stack the coordinates and their derivatives
    x_cartesian = np.stack((x_cart, y_cart), axis=-1)
    dx_cartesian = np.stack((x_dot, y_dot), axis=-1)
    ddx_cartesian = np.stack((x_ddot, y_ddot), axis=-1)

    return x_cartesian, dx_cartesian, ddx_cartesian

# Function to plot the polar spring system data
def plot_pss(data, t, n_ics):
    # Extract the x-coordinate data
    x = data['x']
    n_steps = t.size

    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i in range(n_ics):
        start_idx = i * n_steps
        end_idx = (i + 1) * n_steps

        # Plot theta and r trajectories
        axes[0].plot(t, data["x"][start_idx:end_idx, 0], label=f"IC {i + 1}")
        axes[0].set_title("Theta trajectories")
        axes[0].legend()
        axes[1].plot(t, data["x"][start_idx:end_idx, 1], label=f"IC {i + 1}")
        axes[1].set_title("R trajectories")
        axes[1].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# Function to compare two sets of polar spring system data
def plot_pss_compar(t, x_1, x_2, name_1=None, name_2=None, coord='polar', title=None):
    # Determine the number of features to plot
    num_features = x_1.shape[1]

    # Set up the plot
    fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5))
    if num_features == 1:
        axes = [axes]

    # Set the plot title if provided
    if title is not None:
        fig.suptitle(title)

    # Plot each feature
    for i in range(num_features):
        axes[i].plot(t, x_1[:, i], label=name_1 if name_1 is not None else "x_1")
        if x_2 is not None:
            axes[i].plot(t, x_2[:, i], label=name_2 if name_2 is not None else "x_2")

        # Set titles for each subplot
        if title == 'Latent Space':
            axes[i].set_title(f"Latent dim {i+1}")
        else:
            if coord == 'polar':
                if i == 0:
                    axes[i].set_title("Theta trajectory")
                else:
                    axes[i].set_title(f"R dim {i} trajectory")
            elif coord == 'cartesian':
                axes[i].set_title(f"{chr(88+i)} trajectory")

        axes[i].legend()

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Function to animate the polar spring system data
def animate_pss(data, total_time, t, normalization=1/10):
    # Normalize theta and r values from the dataset
    theta = data['x'][:, 0] * (1 / normalization)
    r = data['x'][:, 1] * (1 / normalization)

    # Create a figure for animation
    fig_anim = plt.figure(figsize=(6, 6))
    ax_anim = fig_anim.add_subplot(111)

    # Set the limits and aspect of the animation plot
    ax_anim.set_xlim(-2.0, 2.0)
    ax_anim.set_ylim(-0.5, 3.0)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)

    # Initialize plot elements for mass, arm line, and base line
    mass, = ax_anim.plot([], [], 'bo-', lw=2)
    arm_line, = ax_anim.plot([], [], 'r-', lw=2)
    base_line, = ax_anim.plot([], [], 'g-', lw=2)

    # Define the update function for animation
    def update(i):
        # Calculate x and y coordinates for mass
        x = [r[i] * np.cos(theta[i] + np.pi / 2)]
        y = [r[i] * np.sin(theta[i] + np.pi / 2)]

        # Update the mass position
        mass.set_data(x, y)

        # Calculate and update the arm line position
        arm_x = np.linspace(0, r[i] * np.cos(theta[i] + np.pi / 2), 10)
        arm_y = np.linspace(0, r[i] * np.sin(theta[i] + np.pi / 2), 10)
        arm_line.set_data(arm_x, arm_y)

        # Update the base line position (static in this case)
        base_x = np.linspace(-0.2, 0.2, 10)
        base_y = np.zeros(10)
        base_line.set_data(base_x, base_y)

        # Return the updated plot elements
        return mass, arm_line, base_line

    # Calculate the interval for the animation frames
    interval = total_time / len(t) * 1000

    # Create the animation
    ani = FuncAnimation(fig_anim, update, frames=len(t), interval=interval, blit=True)

    # Display the animation
    plt.show()

# Main function
def main():
    # Define the number of initial conditions
    n_ics = 10

    # Create an array of time steps from 0 to 20 with a step of 0.02
    t = np.arange(0, 20, .02)

    # Set the coordinate system to Cartesian. Change to 'polar' for polar coordinates
    coord = 'cartesian'

    # Flag to indicate whether to use real dynamics or simulated
    real = True

    # Generate the polar spring system data
    data = get_pss_data(n_ics, t, coord=coord, real=real, noise_strength=0)

    # Plot the polar spring system data
    plot_pss(data, t, n_ics)

    # If the coordinate system is polar, animate the polar spring system data
    if coord == 'polar':
        animate_pss(data, 20, t)

# This condition checks if this script is executed as the main program and not imported as a module
if __name__ == "__main__":
    main()