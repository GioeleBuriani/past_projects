import csv
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

# Function to create and preprocess a dataset from a robot data file
def create_go1_dataset(file_path, normalization=None, apply_filter=False, window_length=5, polyorder=3):
    # Parse initial data from the provided file path
    data = parse_robot_data(file_path)

    # Compute the total force on the Center of Mass (CoM) based on position, control, and foot contact data
    force_on_com = compute_total_force_on_com(data['x'][:, :12], data['u'], data['foot'])
    # Augment control input data with computed total force to match the state's dimensionality
    data['u'] = np.hstack((data['u'], force_on_com))

    # Apply Savitzky-Golay filter to smooth the data if filtering is requested
    if apply_filter:
        for key in ['x', 'dx', 'ddx', 'u']:
            for idx in range(data[key].shape[1]):
                data[key][:, idx] = savgol_filter(data[key][:, idx], window_length, polyorder)

    # Normalize the dataset if a normalization factor is provided
    if normalization is not None:
        for key in ['x', 'dx', 'ddx', 'u']:
            data[key] *= normalization

    return data

# Function to parse robot data from a CSV file
def parse_robot_data(file_path):
    # Initialize lists to hold column indices for different types of data
    x_indices, dx_indices, ddx_indices, u_indices, foot_indices = [], [], [], [], []

    # Open the file and read the headers to classify the data columns
    with open(file_path, 'r') as file:
        headers = file.readline().strip().split(';')
        # Classify headers and store appropriate indices
        for idx, header in enumerate(headers):
            if "Time" in header:
                time_index = idx
            if "Position" in header:
                x_indices.append(idx)
            elif "Velocity" in header:
                dx_indices.append(idx)
            elif "Acceleration" in header:
                ddx_indices.append(idx)
            elif "Torque" in header:
                u_indices.append(idx)
            elif "Foot" in header:
                foot_indices.append(idx)

    # Prepare data dictionary to organize data by type
    data = {'t': [], 'x': [], 'dx': [], 'ddx': [], 'u': [], 'foot': []}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip header row to start reading data
        
        # Read and convert each row into appropriate numeric types and store in the data dictionary
        for row in reader:
            data['t'].append(float(row[time_index].replace(',', '.')))
            data['x'].append([float(row[i].replace(',', '.')) for i in x_indices])
            data['dx'].append([float(row[i].replace(',', '.')) if i < len(row) else None for i in dx_indices])
            data['ddx'].append([float(row[i].replace(',', '.')) if i < len(row) else None for i in ddx_indices])
            data['u'].append([float(row[i].replace(',', '.')) if i < len(row) else None for i in u_indices])
            data['foot'].append([int(float(row[i].replace(',', '.'))) if i < len(row) else None for i in foot_indices])

    # Convert all lists in the dictionary to numpy arrays for better data manipulation and numerical analysis
    for key in data:
        data[key] = np.array(data[key])

    return data

# Function to compute total forces and torques on the Center of Mass (CoM) based on joint configurations
def compute_total_force_on_com(q, tau, contacts):
    # Constants for robotic limb dimensions
    HIP_LINK_LENGTH = 0.0847
    THIGH_LINK_LENGTH = 0.213
    CALF_LINK_LENGTH = 0.213
    NUM_LEGS = 4  # Number of legs on the robot
    LEG_JOINTS = 3  # Number of joints per leg

    n_steps = q.shape[0]  # Number of timesteps in the data
    total_linear_force = np.zeros((n_steps, 3))  # Initialize array to store total linear force per timestep
    total_rotational_force = np.zeros((n_steps, 3))  # Initialize array to store total rotational force per timestep

    # Iterate through each timestep to calculate forces and torques
    for t in range(n_steps):
        linear_force_at_t = np.zeros(3)  # Reset linear force for this timestep
        rotational_force_at_t = np.zeros(3)  # Reset rotational force for this timestep

        # Calculate forces and torques for each leg
        for legID in range(NUM_LEGS):
            # Extract joint angles and torques for the current leg at this timestep
            q_leg = q[t, legID * LEG_JOINTS: (legID + 1) * LEG_JOINTS]
            tau_leg = tau[t, legID * LEG_JOINTS: (legID + 1) * LEG_JOINTS]

            # Compute Jacobian and foot position for the current leg configuration
            J, pos = compute_jacobian(q_leg, legID, HIP_LINK_LENGTH, THIGH_LINK_LENGTH, CALF_LINK_LENGTH)

            # Calculate the force at the foot based on the Jacobian and joint torques
            linear_force_leg = - np.linalg.inv(J.T).dot(tau_leg) * contacts[t, legID]
            rotational_force_leg = np.cross(pos, linear_force_leg)

            # Add calculated forces to the total forces for this timestep
            linear_force_at_t += linear_force_leg
            rotational_force_at_t += rotational_force_leg

        # Store computed total forces and torques for this timestep
        total_linear_force[t, :] = linear_force_at_t
        total_rotational_force[t, :] = rotational_force_at_t

    # Concatenate linear and rotational forces for output
    total_force = np.hstack((total_linear_force, total_rotational_force))

    # Return the total forces on the CoM for each timestep
    return total_force

# Function to compute the Jacobian matrix for a robot leg based on current joint angles
def compute_jacobian(q, legID, l1, l2, l3):
    # Determine the sign multiplier based on leg ID (for symmetry in robot design)
    sideSign = -1 if legID == 0 or legID == 2 else 1

    # Trigonometric calculations for joint angles
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    s3 = np.sin(q[2])
    c1 = np.cos(q[0])
    c2 = np.cos(q[1])
    c3 = np.cos(q[2])
    c23 = c2 * c3 - s2 * s3  # Cosine of combined angle for joints 2 and 3
    s23 = s2 * c3 + c2 * s3  # Sine of combined angle for joints 2 and 3

    # Construct the Jacobian matrix using robot geometry and trigonometry
    J = np.zeros((3, 3))
    J[1, 0] = -sideSign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
    J[2, 0] = sideSign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
    J[0, 1] = -l3 * c23 - l2 * c2
    J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
    J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
    J[0, 2] = -l3 * c23
    J[1, 2] = -l3 * s23 * s1
    J[2, 2] = l3 * s23 * c1

    # Calculate the position of the foot based on the joint angles and link lengths
    pos = np.zeros(3)
    pos[0] = -l3 * s23 - l2 * s2
    pos[1] = l1 * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
    pos[2] = l1 * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

    return J, pos

# Function to plot joint data and compare with learned joint data if available
def plot_joint_data(time, joint_data, learned_joint_data=None):
    # Calculate the global minimum and maximum from the joint data to set y-axis limits
    global_min = joint_data.min()
    global_max = joint_data.max()

    # If learned data is provided, adjust the global min and max to include its values
    if learned_joint_data is not None:
        global_min = min(global_min, learned_joint_data.min())
        global_max = max(global_max, learned_joint_data.max())

    # Define a margin to provide some padding around the plotted data
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots to plot each joint
    fig, axs = plt.subplots(3, 4, figsize=(16, 10))  # 3 rows and 4 columns of plots

    # Loop through the subplots and plot data for each joint
    for i in range(3):
        for j in range(4):
            joint_idx = i + j*3
            axs[i, j].plot(time, joint_data[:, joint_idx], label=f'Original Joint {joint_idx+1}')
            if learned_joint_data is not None:
                axs[i, j].plot(time, learned_joint_data[:, joint_idx], label=f'Learned Joint {joint_idx+1}')
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('Position')
            axs[i, j].set_ylim(y_min, y_max)  # Ensure all plots have the same y-axis limits
            axs[i, j].legend()

    # Set a main title for the entire figure
    fig.suptitle('Joint State Over Time', fontsize=16)

    # Adjust layout to ensure labels and titles do not overlap
    plt.tight_layout()
    plt.show(block=False)

# Function to plot body data and compare with learned body data if available
def plot_body_data(time, body_data, learned_body_data=None):
    # Calculate the global minimum and maximum from the body data to set y-axis limits
    global_min = body_data.min()
    global_max = body_data.max()

    # If learned data is provided, adjust the global min and max to include its values
    if learned_body_data is not None:
        global_min = min(global_min, learned_body_data.min())
        global_max = max(global_max, learned_body_data.max())

    # Define a margin to provide some padding around the plotted data
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots to plot each body state
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows and 3 columns of plots

    # Flatten the array of axes for easy indexing
    axs = axs.flatten()

    # Loop through the subplots and plot data for each body state
    for i in range(6):
        axs[i].plot(time, body_data[:, i], label=f'Body {i+1}')
        if learned_body_data is not None:
            axs[i].plot(time, learned_body_data[:, i], label=f'Learned Body {i+1}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Value')
        axs[i].set_ylim(y_min, y_max)  # Ensure all plots have the same y-axis limits
        axs[i].legend()

    # Set a main title for the entire figure
    fig.suptitle('Body State Over Time', fontsize=16)

    # Adjust layout to ensure labels and titles do not overlap
    plt.tight_layout()
    plt.show(block=False)

# Function to plot torque data for different joints along with optionally provided learned torque data
def plot_torque_data(time, torque_data, learned_torque_data=None):
    # Define labels for each torque dimension
    labels = ['fr.hx Torque', 'fr.hy Torque', 'fr.kn Torque', 'fl.hx Torque', 'fl.hy Torque', 'fl.kn Torque',
              'rr.hx Torque', 'rr.hy Torque', 'rr.kn Torque', 'rl.hx Torque', 'rl.hy Torque', 'rl.kn Torque']

    # Calculate global minimum and maximum values for y-axis limits from the torque data
    global_min = torque_data.min()
    global_max = torque_data.max()

    # If learned torque data is provided, adjust global min and max accordingly
    if learned_torque_data is not None:
        global_min = min(global_min, learned_torque_data.min())
        global_max = max(global_max, learned_torque_data.max())

    # Set a margin to ensure that the plot does not cut off data extremities
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots to accommodate all torque dimensions
    fig, axs = plt.subplots(3, 4, figsize=(16, 10))  # 3 rows and 4 columns

    # Plot each torque dimension in a separate subplot
    for i in range(3):
        for j in range(4):
            joint_idx = i + j*3
            axs[i, j].plot(time, torque_data[:, joint_idx], label=labels[joint_idx])
            if learned_torque_data is not None:
                axs[i, j].plot(time, learned_torque_data[:, joint_idx], label='Learned '+labels[joint_idx])
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('Torque')
            axs[i, j].set_ylim(y_min, y_max)  # Apply consistent y-axis limits
            axs[i, j].legend()

    # Set a main title for the entire figure
    fig.suptitle('Joint Torque Over Time', fontsize=16)

    # Adjust subplot layout to ensure labels and titles do not overlap
    plt.tight_layout()
    plt.show(block=False)

# Function to plot force data on the Center of Mass (CoM) over time
def plot_forces_on_com(time, force_data):
    # Calculate global minimum and maximum values for y-axis limits from the force data
    global_min = force_data.min()
    global_max = force_data.max()

    # Set a margin to ensure that the plot does not cut off data extremities
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots to accommodate all force dimensions
    fig, axs = plt.subplots(2, 3, figsize=(16, 5))  # 2 rows and 3 columns

    # Plot each force dimension in a separate subplot
    for i in range(6):
        ax = axs[i // 3, i % 3]  # Map the 1D index to 2D grid position
        ax.plot(time, force_data[:, i], label=f'Force {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force')
        ax.set_ylim(y_min, y_max)  # Apply consistent y-axis limits
        ax.legend()

    # Set a main title for the entire figure
    fig.suptitle('Force on CoM Over Time', fontsize=16)

    # Adjust subplot layout to ensure labels and titles do not overlap
    plt.tight_layout()
    plt.show(block=False)

# Function to plot data showing foot contact over time for each foot
def plot_foot_contact_data(time, foot_contact_data):
    # Initialize a figure with 2x2 grid of subplots for displaying foot contact data
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Iterate over each subplot position to plot corresponding foot contact data
    for i in range(2):
        for j in range(2):
            foot_idx = i * 2 + j  # Calculate index for the foot data array
            axs[i, j].plot(time, foot_contact_data[:, foot_idx], label=f'Foot {foot_idx + 1}')
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('Foot Contact')
            axs[i, j].legend()  # Add a legend to each subplot

    # Set the title for the entire figure
    fig.suptitle('Foot Contacts Over Time', fontsize=16)

    # Adjust layout to prevent overlapping of subplot elements
    plt.tight_layout()
    plt.show(block=False)

# Function to plot comparison data for two datasets over time
def plot_go1_compar(t, x_1, x_2, name_1=None, name_2=None, coord='polar', title=None):
    # Determine the number of features in the datasets
    num_features = x_1.shape[1]
    # Define the layout of the subplots
    num_columns = 4
    subplot_width = 5
    subplot_height = 4
    # Calculate the number of rows needed to display all features
    num_rows = -(-num_features // num_columns)

    # Create a figure with the calculated number of rows and columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * subplot_width, num_rows * subplot_height))

    # Convert axes array to 2D if only one row of subplots is present
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)

    # Remove unused axes if there are more subplots than features
    for i in range(num_features, num_rows * num_columns):
        fig.delaxes(axes.flatten()[i])

    # Set the title for the figure if specified
    if title is not None:
        fig.suptitle(title)

    # Plot each feature in its respective subplot
    for i in range(num_features):
        row, col = divmod(i, num_columns)
        axes[row, col].plot(t, x_1[:, i], label=name_1 if name_1 is not None else "x_1")

        # Plot secondary dataset if provided
        if x_2 is not None:
            axes[row, col].plot(t, x_2[:, i], label=name_2 if name_2 is not None else "x_2")

        # Set individual subplot titles for special cases like 'Latent Space'
        if title == 'Latent Space':
            axes[row, col].set_title(f"Latent dim {i + 1}")

        axes[row, col].legend()

    # Adjust the layout of the subplots to fit better within the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)

# Main function to execute the data processing and plotting for robot simulation data
def main():
    # Define the file path for the dataset
    file_path = "go1_sim/prove/prova4_fixed.csv"
    
    # Create a dataset from the specified file without applying normalization or filtering
    data = create_go1_dataset(file_path, normalization=None, apply_filter=False)

    # Restrict the dataset to a specific range of indices for focused analysis
    data = {key: data[key][800:1600] for key in data}

    # Print the dimensions of various data components to verify correct data handling
    print(f"Time: {data['t'].shape}")
    print(f"Position: {data['x'].shape}")
    print(f"Velocity: {data['dx'].shape}")
    print(f"Acceleration: {data['ddx'].shape}")
    print(f"Control: {data['u'].shape}")
    print(f"Foot Contact: {data['foot'].shape}")

    ### JOINT DATA ###
    # Extract joint-related data for plotting
    time = data['t']
    joint_positions = data['x'][:, 0:12]
    joint_velocities = data['dx'][:, 0:12]
    joint_accelerations = data['ddx'][:, 0:12]
    # Plot joint position data
    plot_joint_data(time, joint_positions)

    ### BODY DATA ###
    # Extract body-related data for plotting
    time = data['t']
    body_positions = data['x'][:, 12:18]
    body_velocities = data['dx'][:, 12:18]
    body_accelerations = data['ddx'][:, 12:18]
    # Plot body position data
    plot_body_data(time, body_positions)

    ### TORQUE DATA ###
    # Extract torque-related data for plotting
    time = data['t']
    torque_data = data['u'][:, 0:12]
    force_data = data['u'][:, 12:18]
    # Plot torque data and forces on the center of mass
    plot_torque_data(time, torque_data)
    plot_forces_on_com(time, force_data)

    ### FOOT CONTACT DATA ###
    # Extract foot contact data for plotting
    time = data['t']
    foot_contact_data = data['foot'][:, 0:4]
    # Plot foot contact data
    plot_foot_contact_data(time, foot_contact_data)

    # Show all plots generated from the data
    plt.show()

# Execute the main function if this script is run as the main program
if __name__ == "__main__":
    main()