import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set a fixed seed for reproducibility of simulation outcomes
SEED = 0
np.random.seed(SEED)

# Function to generate Double Pendulum Spring System (DPSS) dataset
def get_dpss_data(n_ics, t, coord='polar', real=False, input_type=None, noise_strength=0):

    # Determine the number of time steps and input dimensions
    n_steps = t.size
    input_dim = 4

    # Initialize random values for initial conditions within specified ranges
    theta_10 = np.random.uniform(-np.pi/2, np.pi/2, n_ics)
    r_10 = np.random.uniform(0.5, 2.5, n_ics)
    theta_20 = np.random.uniform(-np.pi/2, np.pi/2, n_ics)
    r_20 = np.random.uniform(0.5, 2.5, n_ics)
    theta_1_dot0 = np.random.uniform(-0.5, 0.5, n_ics)
    r_1_dot0 = np.random.uniform(-0.5, 0.5, n_ics)
    theta_2_dot0 = np.random.uniform(-0.5, 0.5, n_ics)
    r_2_dot0 = np.random.uniform(-0.5, 0.5, n_ics)
    
    # Stack all initial conditions into a single array
    ics = np.column_stack((theta_10, r_10, theta_20, r_20, theta_1_dot0, r_1_dot0, theta_2_dot0, r_2_dot0))

    # Generate data based on the initial conditions and simulation parameters
    data = generate_dpss_data(ics, t, normalization=np.array([1/10, 1/10, 1/10, 1/10]), coord=coord, real=real, input_type=input_type)

    # Introduce Gaussian noise to the dataset to simulate measurement errors
    data['x'] += noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['dx'] += noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['ddx'] += noise_strength * np.random.randn(n_steps * n_ics, input_dim)
    data['u'] += noise_strength * np.random.randn(n_steps * n_ics, input_dim)

    return data

# Function to simulate the dynamics of a double pendulum system
def generate_dpss_data(ics, t, normalization=None, coord='polar', real=False, input_type=None,
                       k_arm_1=50.0, b_arm_1=0.5,
                       k_angle_1=50.0, b_angle_1=0.5,
                       k_arm_2=5000.0, b_arm_2=50.0,
                       k_angle_2=5000.0, b_angle_2=50.0,
                       m=1.0, arm_length=1.5):

    # Determine the number of initial conditions and simulation steps
    n_ics = ics.shape[0]
    n_steps = t.size

    # Initialize arrays to hold the state variables and their derivatives
    d = 4  # System dimensionality
    x = np.zeros((n_ics, n_steps, d))
    dx = np.zeros(x.shape)
    ddx = np.zeros(x.shape)
    u = np.zeros(x.shape)  # Control inputs for each time step

    # Simulate each set of initial conditions
    for i in range(n_ics):
        x[i], dx[i], ddx[i], u[i] = simulate_dpss(ics[i], t, input_type=input_type,
                                                  k_arm_1=k_arm_1, b_arm_1=b_arm_1,
                                                  k_angle_1=k_angle_1, b_angle_1=b_angle_1,
                                                  k_arm_2=k_arm_2, b_arm_2=b_arm_2,
                                                  k_angle_2=k_angle_2, b_angle_2=b_angle_2,
                                                  m=m, arm_length=arm_length)

    # Optionally calculate derivatives directly from simulated positions if 'real' flag is set
    if real:
        dx = np.gradient(x, t, axis=1)
        ddx = np.gradient(dx, t, axis=1)

    # Apply normalization factors if provided
    if normalization:
        x *= normalization
        dx *= normalization
        ddx *= normalization

    # Convert from polar to Cartesian coordinates if required
    if coord == 'cartesian':
        x, dx, ddx = polar_to_cartesian(x, dx, ddx)

    # Package the simulation results into a dictionary for easy access
    data = {'t': t, 'x': x, 'dx': dx, 'ddx': ddx, 'u': u}

    return data

# Function to simulate dynamics for a double pendulum spring system (DPSS)
def simulate_dpss(x0, t, input_type=None,
                  k_arm_1=5000.0, b_arm_1=50.0,
                  k_angle_1=50.0, b_angle_1=0.5,
                  k_arm_2=5000.0, b_arm_2=50.0,
                  k_angle_2=5000.0, b_angle_2=50.0,
                  m=1.0, arm_length=1.5):
    
    g = 9.81  # Gravitational acceleration

    # Nested function to calculate the dynamics at each time step
    def f(y, t):
        # Unpack the current state of the system
        theta_1, r_1, theta_2, r_2, theta_1_dot, r_1_dot, theta_2_dot, r_2_dot = y

        # Initialize control inputs
        u_1, u_2 = 0, 0

        # Determine control inputs based on specified input type
        if input_type == 'constant':
            u_1, u_2 = 10, 10
        elif input_type == 'sinusoidal':
            u_1, u_2 = np.sin(t) * 10, np.sin(t) * 10
        elif input_type == 'P':
            # Proportional control based on the error in angles
            error_theta_1 = -theta_1
            error_theta_2 = -theta_2
            u_1 = 10 * error_theta_1
            u_2 = 10 * error_theta_2

        # Calculate the accelerations based on the current state and control inputs
        theta_1_ddot = g * r_1 * np.sin(theta_1) + g * (r_2 / 2) * np.sin(theta_1) \
                       - k_angle_1 * theta_1 / (2 * m) - b_angle_1 * theta_1_dot / (2 * m) + u_1
        r_1_ddot = -g * np.cos(theta_1) - k_arm_1 * (r_1 - arm_length) / (2 * m) - b_arm_1 * r_1_dot / (2 * m)
        theta_2_ddot = g * r_2 * np.sin(theta_1 + theta_2) - k_angle_2 * theta_2 / m \
                       - b_angle_2 * theta_2_dot / m + u_2
        r_2_ddot = -g * np.cos(theta_1 + theta_2) - k_arm_2 * (r_2 - arm_length) / m - b_arm_2 * r_2_dot / m

        return [theta_1_dot, r_1_dot, theta_2_dot, r_2_dot, theta_1_ddot, r_1_ddot, theta_2_ddot, r_2_ddot]
    
    # Integrate the differential equations over the time array t
    sol = odeint(f, x0, t)
    
    # Extract positions and their first derivatives
    x = sol[:, 0:4]
    dx = sol[:, 4:8]
    
    # Calculate second derivatives and control inputs dynamically for each time step
    ddx = np.zeros_like(x)
    u = np.zeros((t.size, 2))  # Initialize storage for control inputs
    for i in range(t.size):
        ddx[i] = f(sol[i], t[i])[4:8]
        # Re-calculate control inputs based on the dynamics function
        u[i] = [f(sol[i], t[i])[4], f(sol[i], t[i])[6]]

    return x, dx, ddx, u

# Function to convert polar coordinates to Cartesian coordinates
def polar_to_cartesian(x, dx, ddx):

    # Extract components from polar coordinates
    theta_1, r_1 = x[:,:,0], x[:,:,1]
    theta_1_dot, r_1_dot = dx[:,:,0], dx[:,:,1]
    theta_1_ddot, r_1_ddot = ddx[:,:,0], ddx[:,:,1]

    theta_2, r_2 = x[:,:,2], x[:,:,3]
    theta_2_dot, r_2_dot = dx[:,:,2], dx[:,:,3]
    theta_2_ddot, r_2_ddot = ddx[:,:,2], ddx[:,:,3]

    # Convert polar coordinates to Cartesian coordinates for the first pendulum
    x1_cart = r_1 * np.sin(theta_1)
    y1_cart = r_1 * np.cos(theta_1)
    x1_dot = r_1_dot * np.sin(theta_1) + r_1 * theta_1_dot * np.cos(theta_1)
    y1_dot = r_1_dot * np.cos(theta_1) - r_1 * theta_1_dot * np.sin(theta_1)
    x1_ddot = r_1_ddot * np.sin(theta_1) + 2 * r_1_dot * theta_1_dot * np.cos(theta_1) - r_1 * theta_1_dot**2 * np.sin(theta_1)
    y1_ddot = r_1_ddot * np.cos(theta_1) - 2 * r_1_dot * theta_1_dot * np.sin(theta_1) - r_1 * theta_1_dot**2 * np.cos(theta_1)

    # Convert polar coordinates to Cartesian coordinates for the second pendulum, dependent on the first pendulum's endpoint
    x2_cart = x1_cart + r_2 * np.sin(theta_1 + theta_2)
    y2_cart = y1_cart + r_2 * np.cos(theta_1 + theta_2)
    x2_dot = x1_dot + r_2_dot * np.sin(theta_1 + theta_2) + r_2 * (theta_1_dot + theta_2_dot) * np.cos(theta_1 + theta_2)
    y2_dot = y1_dot + r_2_dot * np.cos(theta_1 + theta_2) - r_2 * (theta_1_dot + theta_2_dot) * np.sin(theta_1 + theta_2)
    x2_ddot = x1_ddot + r_2_ddot * np.sin(theta_1 + theta_2) + 2 * r_2_dot * np.cos(theta_1 + theta_2) * (theta_1_dot + theta_2_dot) + r_2 * np.cos(theta_1 + theta_2) * (theta_1_ddot + theta_2_ddot) - r_2 * np.sin(theta_1 + theta_2) * (theta_1_dot + theta_2_dot)**2
    y2_ddot = y1_ddot + r_2_ddot * np.cos(theta_1 + theta_2) - 2 * r_2_dot * np.sin(theta_1 + theta_2) * (theta_1_dot + theta_2_dot) - r_2 * np.sin(theta_1 + theta_2) * (theta_1_ddot + theta_2_ddot) - r_2 * np.cos(theta_1 + theta_2) * (theta_1_dot + theta_2_dot)**2

    # Stack coordinates for easy access and manipulation
    x_cartesian = np.stack((x1_cart, y1_cart, x2_cart, y2_cart), axis=-1)
    dx_cartesian = np.stack((x1_dot, y1_dot, x2_dot, y2_dot), axis=-1)
    ddx_cartesian = np.stack((x1_ddot, y1_ddot, x2_ddot, y2_ddot), axis=-1)

    return x_cartesian, dx_cartesian, ddx_cartesian

# Function to plot the dynamics of the double pendulum spring system (DPSS)
def plot_dpss(t, x, name=None, coord='polar', title=None):

    num_features = x.shape[1]  # Determine the number of state variables to plot
    
    # Layout parameters for the plot
    num_columns = 2
    subplot_width = 5
    subplot_height = 4
    num_rows = -(-num_features // num_columns)  # Compute the number of rows needed
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * subplot_width, num_rows * subplot_height))
    
    # Ensure axes is always a 2D array for consistent indexing
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Remove unused axes if the number of features is less than the number of subplots
    for i in range(num_features, num_rows * num_columns):
        fig.delaxes(axes.flatten()[i])

    # Set the title for the entire figure if provided
    if title is not None:
        fig.suptitle(title)

    # Plot each feature in its subplot
    for i in range(num_features):
        row, col = divmod(i, 2)
        axes[row, col].plot(t, x[:, i], label=name if name is not None else "x")
        
        # Customize the subplot title based on the type of trajectory and coordinate system
        if title == 'Latent Space':
            axes[row, col].set_title(f"Latent dim {i+1}")
        else:
            if coord == 'polar':
                axes[row, col].set_title(f"Theta{i//2 + 1} trajectory" if i % 2 == 0 else f"R{i//2 + 1} trajectory")
            elif coord == 'cartesian':
                axes[row, col].set_title(f"{chr(88+i)} trajectory")

        axes[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)

# Function to compare the trajectories of two different simulations of the DPSS
def plot_dpss_compar(t, x_1, x_2, name_1=None, name_2=None, coord='polar', title=None):

    num_features = x_1.shape[1]  # Determine the number of state variables
    
    # Layout parameters for the plot
    num_columns = 2
    subplot_width = 5
    subplot_height = 4
    num_rows = -(-num_features // num_columns)  # Compute the number of rows needed
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * subplot_width, num_rows * subplot_height))
    
    # Ensure axes is always a 2D array for consistent indexing
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Remove unused axes if the number of features is less than the number of subplots
    for i in range(num_features, num_rows * num_columns):
        fig.delaxes(axes.flatten()[i])

    # Set the title for the entire figure if provided
    if title is not None:
        fig.suptitle(title)

    # Plot each feature in its subplot, comparing two different data sets
    for i in range(num_features):
        row, col = divmod(i, 2)
        axes[row, col].plot(t, x_1[:, i], label=name_1 if name_1 is not None else "x_1")
        axes[row, col].plot(t, x_2[:, i], label=name_2 if name_2 is not None else "x_2")

        # Customize the subplot title based on the type of trajectory and coordinate system
        if title == 'Latent Space':
            axes[row, col].set_title(f"Latent dim {i+1}")
        else:
            if coord == 'polar':
                axes[row, col].set_title(f"Theta{i//2 + 1} trajectory" if i % 2 == 0 else f"R{i//2 + 1} trajectory")
            elif coord == 'cartesian':
                axes[row, col].set_title(f"{chr(88+i)} trajectory")

        axes[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)

# Function to animate the double pendulum spring system (DPSS) dynamics
def animate_dpss(data, total_time, t, normalization=1/10):

    # Normalize and convert polar coordinates to Cartesian coordinates
    theta_1 = data['x'][:, 0] * (1 / normalization)
    r_1 = data['x'][:, 1] * (1 / normalization)
    theta_2 = data['x'][:, 2] * (1 / normalization)
    r_2 = data['x'][:, 3] * (1 / normalization)

    x_1 = r_1 * np.sin(theta_1)  # X position of the first pendulum
    y_1 = r_1 * np.cos(theta_1)  # Y position of the first pendulum

    x_2 = x_1 + r_2 * np.sin(theta_1 + theta_2)  # X position of the second pendulum
    y_2 = y_1 + r_2 * np.cos(theta_1 + theta_2)  # Y position of the second pendulum

    # Set up the plot for animation
    fig_anim = plt.figure(figsize=(6, 6))
    ax_anim = fig_anim.add_subplot(111, autoscale_on=False, xlim=(-4, 4), ylim=(-0.5, 5))
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)

    # Initialize the lines that will be updated during the animation
    masses, = ax_anim.plot([], [], 'bo-', lw=2)  # Masses of pendulums
    arm_line_1, = ax_anim.plot([], [], 'r-', lw=2)  # Arm of the first pendulum
    arm_line_2, = ax_anim.plot([], [], 'b-', lw=2)  # Arm of the second pendulum
    base_line, = ax_anim.plot([], [], 'g-', lw=2)  # Base line

    # Define the update function for animation
    def update(i):
        # Set data for each frame of the animation
        masses.set_data([0, x_1[i], x_2[i]], [0, y_1[i], y_2[i]])
        arm_line_1.set_data([0, x_1[i]], [0, y_1[i]])
        arm_line_2.set_data([x_1[i], x_2[i]], [y_1[i], y_2[i]])
        
        # Draw a base line
        base_x = np.linspace(-0.2, 0.2, 10)
        base_y = np.zeros(10)
        base_line.set_data(base_x, base_y)

        return masses, arm_line_1, arm_line_2, base_line

    # Set the interval for frames based on the total time and number of time steps
    interval = total_time / len(t) * 1000  # Convert to milliseconds

    # Create and start the animation
    ani = FuncAnimation(fig_anim, update, frames=len(t), interval=interval, blit=True)

    plt.show()

# Main function to run the DPSS simulation and animation
def main():
    # Number of initial conditions
    n_ics = 1
    # Generate time points
    t = np.arange(0, 20, .01)

    # Coordinate system for the simulation
    coord = 'polar'
    
    # Generate simulation data
    data = get_dpss_data(n_ics, t, coord=coord, input_type='sinusoidal', noise_strength=0)
    
    # Plot static trajectories of the DPSS
    plot_dpss(t, data['x'], n_ics)

    # Animate the dynamic behavior of the DPSS
    animate_dpss(data, 20, t)

# Execute the main function
if __name__ == "__main__":
    main()