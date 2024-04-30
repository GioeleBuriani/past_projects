import csv
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

# Function to create a dataset from a robot data file
def create_a1_dataset(file_path, normalization=None, apply_filter=False, window_length=5, polyorder=3):
    # Parse data from the file
    data = parse_robot_data(file_path)

    # If filtering is requested, apply Savitzky-Golay filter to certain data types
    if apply_filter:
        # Loop through data types for filtering
        for key in ['x', 'dx', 'ddx', 'u']:
            # Apply filter to each dimension of the data type
            for idx in range(data[key].shape[1]):
                data[key][:, idx] = savgol_filter(data[key][:, idx], window_length, polyorder)

    # Normalize data if a normalization factor is provided
    if normalization is not None:
        # Multiply each data type by the normalization factor
        for key in ['x', 'dx', 'ddx', 'u']:
            data[key] *= normalization

    return data

# Function to parse robot data from a CSV file
def parse_robot_data(file_path):
    # Initialize lists to hold column indices for different types of robot data
    x_indices, dx_indices, ddx_indices, u_indices, foot_indices = [], [], [], [], []

    # Open the file and read the headers to determine the structure of the data
    with open(file_path, 'r') as file:
        headers = file.readline().strip().split(';')
    
        # Identify and store the appropriate indices for each type of data based on header labels
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
            elif "Force" in header:
                u_indices.append(idx)
            elif "Foot" in header:
                foot_indices.append(idx)

    # Create a dictionary to store the parsed data, organized by type
    data = {'t': [], 'x': [], 'dx': [], 'ddx': [], 'u': [], 'foot': []}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip the header row

        # Read each row, convert string to appropriate numeric type, and append to corresponding list
        for row in reader:
            data['t'].append(float(row[time_index].replace(',', '.')))
            data['x'].append([float(row[i].replace(',', '.')) for i in x_indices])
            data['dx'].append([float(row[i].replace(',', '.')) if i < len(row) else None for i in dx_indices])
            data['ddx'].append([float(row[i].replace(',', '.')) if i < len(row) else None for i in ddx_indices])
            data['u'].append([float(row[i].replace(',', '.')) if i < len(row) else None for i in u_indices])
            data['foot'].append([int(float(row[i].replace(',', '.'))) if i < len(row) else None for i in foot_indices])

    # Convert all lists in the dictionary to numpy arrays for better handling in numerical computations
    for key in data:
        data[key] = np.array(data[key])

    return data

# Function to plot data from individual joints along with optionally provided learned joint data
def plot_joint_data(time, joint_data, learned_joint_data=None):
    # Determine the minimum and maximum values for setting y-axis limits
    global_min = joint_data.min()
    global_max = joint_data.max()

    # Update the global minimum and maximum if learned joint data is provided
    if learned_joint_data is not None:
        global_min = min(global_min, learned_joint_data.min())
        global_max = max(global_max, learned_joint_data.max())

    # Set margins for the plot
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots
    fig, axs = plt.subplots(3, 4, figsize=(16, 10))  # 3 rows and 4 columns layout

    # Plot each joint's data in a separate subplot
    for i in range(3):
        for j in range(4):
            joint_idx = i + j*3
            axs[i, j].plot(time, joint_data[:, joint_idx], label=f'Original Joint {joint_idx+1}')
            if learned_joint_data is not None:
                axs[i, j].plot(time, learned_joint_data[:, joint_idx], label=f'Learned Joint {joint_idx+1}')
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('Position')
            axs[i, j].set_ylim(y_min, y_max)  # Apply the calculated y-axis limits consistently across all plots
            axs[i, j].legend()

    # Title for the entire figure
    fig.suptitle('Joint State Over Time', fontsize=16)

    # Adjust subplot layout for clarity
    plt.tight_layout()
    plt.show(block=False)

# Function to plot data from body metrics along with optionally provided learned body data
def plot_body_data(time, body_data, learned_body_data=None):
    # Determine the minimum and maximum values for setting y-axis limits
    global_min = body_data.min()
    global_max = body_data.max()

    # Update the global minimum and maximum if learned body data is provided
    if learned_body_data is not None:
        global_min = min(global_min, learned_body_data.min())
        global_max = max(global_max, learned_body_data.max())

    # Set margins for the plot
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots in a 2x3 configuration
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Flatten the array of axes for easier indexing
    axs = axs.flatten()

    # Plot each body metric's data
    for i in range(6):
        axs[i].plot(time, body_data[:, i], label=f'Body {i+1}')
        if learned_body_data is not None:
            axs[i].plot(time, learned_body_data[:, i], label=f'Learned Body {i+1}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Value')
        axs[i].set_ylim(y_min, y_max)  # Apply the calculated y-axis limits consistently across all plots
        axs[i].legend()

    # Title for the entire figure
    fig.suptitle('Body State Over Time', fontsize=16)

    # Adjust subplot layout for clarity
    plt.tight_layout()
    plt.show(block=False)

# Function to plot the trajectory of the body in X-Y plane
def plot_body_trajectory(body_positions, learned_body_positions=None):
    # Create a plot for the trajectory
    plt.figure(figsize=(8, 8))

    # Plot the original trajectory of the robot
    plt.plot(body_positions[:, 0], body_positions[:, 1], label='Robot Trajectory')

    # Plot the learned trajectory if provided
    if learned_body_positions is not None:
        plt.plot(learned_body_positions[:, 0], learned_body_positions[:, 1], label='Learned Robot Trajectory')

    # Add labels and title
    plt.title('Robot X-Y Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()

    # Set equal aspect ratio
    plt.axis('equal')

    # Display the plot
    plt.show(block=False)

# Function to plot torque data for joints alongside optionally provided learned torque data
def plot_torque_data(time, torque_data, learned_torque_data=None):
    # Labels for individual torques
    labels = ['fr.hx Torque', 'fr.hy Torque', 'fr.kn Torque', 'fl.hx Torque', 'fl.hy Torque', 'fl.kn Torque',
              'rr.hx Torque', 'rr.hy Torque', 'rr.kn Torque', 'rl.hx Torque', 'rl.hy Torque', 'rl.kn Torque']

    # Determine the minimum and maximum values for setting y-axis limits
    global_min = torque_data.min()
    global_max = torque_data.max()

    # Update the global minimum and maximum if learned torque data is provided
    if learned_torque_data is not None:
        global_min = min(global_min, learned_torque_data.min())
        global_max = max(global_max, learned_torque_data.max())

    # Set margins for the plot
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots
    fig, axs = plt.subplots(3, 4, figsize=(16, 10))  # 3 rows and 4 columns layout

    # Plot each torque in a separate subplot
    for i in range(3):
        for j in range(4):
            joint_idx = i + j*3
            axs[i, j].plot(time, torque_data[:, joint_idx], label=labels[joint_idx])
            if learned_torque_data is not None:
                axs[i, j].plot(time, learned_torque_data[:, joint_idx], label='Learned '+labels[joint_idx])
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('Torque')
            axs[i, j].set_ylim(y_min, y_max)  # Apply the calculated y-axis limits consistently across all plots
            axs[i, j].legend()

    # Title for the entire figure
    fig.suptitle('Joint Torque Over Time', fontsize=16)

    # Adjust subplot layout for clarity
    plt.tight_layout()
    plt.show(block=False)

# Function to plot forces on the center of mass (CoM) over time
def plot_forces_on_com(time, force_data):
    # Calculate global minimum and maximum forces to determine y-axis limits
    global_min = force_data.min()
    global_max = force_data.max()

    # Set margins for the plot
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(16, 5))  # 2 rows and 3 columns layout

    # Plot each force component in a separate subplot
    for i in range(6):
        ax = axs[i // 3, i % 3]  # Determine subplot position
        ax.plot(time, force_data[:, i], label=f'Force {i+1}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force')
        ax.set_ylim(y_min, y_max)  # Apply the calculated y-axis limits consistently across all plots
        ax.legend()

    # Title for the entire figure
    fig.suptitle('Force on CoM Over Time', fontsize=16)

    # Adjust subplot layout for clarity
    plt.tight_layout()
    plt.show(block=False)

# Function to plot foot contact data over time
def plot_foot_contact_data(time, foot_contact_data):
    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Plot each foot's contact data in a separate subplot
    for i in range(2):
        for j in range(2):
            foot_idx = i*2 + j
            axs[i, j].plot(time, foot_contact_data[:, foot_idx], label=f'Foot {foot_idx+1}')
            axs[i, j].set_xlabel('Time (s)')
            axs[i, j].set_ylabel('Foot Contact')
            axs[i, j].legend()

    # Title for the entire figure
    fig.suptitle('Foot Contacts Over Time', fontsize=16)

    # Adjust subplot layout to prevent overlap and ensure clarity
    plt.tight_layout()
    plt.show(block=False)

# Function to plot comparisons between two sets of data across multiple features
def plot_a1_compar(t, x_1, x_2, name_1=None, name_2=None, coord='polar', title=None):
    # Calculate the number of features in the data
    num_features = x_1.shape[1]

    # Set up subplot grid dimensions
    num_columns = 4
    subplot_width = 5
    subplot_height = 4
    num_rows = -(-num_features // num_columns)  # Ceil division to get required number of rows

    # Create a figure with a grid of subplots based on the number of features
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns*subplot_width, num_rows*subplot_height))
    
    # Ensure axes is a 2D array for easier indexing, necessary when only one row of subplots
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Remove any unused axes if there are fewer features than subplots
    for i in range(num_features, num_rows*num_columns):
        fig.delaxes(axes.flatten()[i])

    # Set a title for the figure if provided
    if title is not None:
        fig.suptitle(title)

    # Plot each feature in its own subplot
    for i in range(num_features):
        row, col = divmod(i, num_columns)  # Determine the position of the subplot
        axes[row, col].plot(t, x_1[:, i], label=name_1 if name_1 is not None else "x_1")  # Plot first data set
        
        # Plot second data set if it's provided
        if x_2 is not None:
            axes[row, col].plot(t, x_2[:, i], label=name_2 if name_2 is not None else "x_2")

        # Optionally set individual subplot titles for specific cases, like latent dimensions
        if title == 'Latent Space':
            axes[row, col].set_title(f"Latent dim {i+1}")

        # Add a legend to each subplot
        axes[row, col].legend()

    # Adjust layout to prevent subplot labels from overlapping and ensure the title fits
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)



# Main function to execute the data analysis and plotting process
def main():
    # Path to the dataset file
    file_path = "go1_sim/a1_jump_dataset_test.csv"
    
    # Create dataset from the file without normalization or filtering
    data = create_a1_dataset(file_path, normalization=None, apply_filter=False)

    # Slice the dataset to focus on a specific range of indices
    data = {key: data[key][1200:2400] for key in data}

    # Print the shapes of various data components to verify the slicing and loading
    print(f"Time: {data['t'].shape}")
    print(f"Position: {data['x'].shape}")
    print(f"Velocity: {data['dx'].shape}")
    print(f"Acceleration: {data['ddx'].shape}")
    print(f"Control: {data['u'].shape}")
    print(f"Foot Contact: {data['foot'].shape}")

    ### JOINT DATA ###
    # Extract and prepare joint data for plotting
    time = data['t']
    joint_positions = data['x'][:, 0:12]
    joint_velocities = data['dx'][:, 0:12]
    joint_accelerations = data['ddx'][:, 0:12]
    # Call function to plot joint data
    plot_joint_data(time, joint_positions)

    ### BODY DATA ###
    # Extract and prepare body data for plotting
    time = data['t']
    body_positions = data['x'][:, 12:18]
    body_velocities = data['dx'][:, 12:18]
    body_accelerations = data['ddx'][:, 12:18]
    # Call function to plot body data
    plot_body_data(time, body_positions)

    ### TORQUE DATA ###
    # Extract and prepare torque data for plotting
    time = data['t']
    torque_data = data['u'][:, 0:12]
    force_data = data['u'][:, 12:18]
    # Call functions to plot torque and force data
    plot_torque_data(time, torque_data)
    plot_forces_on_com(time, force_data)

    ### FOOT CONTACT DATA ###
    # Extract and prepare foot contact data for plotting
    time = data['t']
    foot_contact_data = data['foot'][:, 0:4]
    # Call function to plot foot contact data
    plot_foot_contact_data(time, foot_contact_data)

    # Show all plots at the end of data processing
    plt.show()

# Execute the main function if this script is run as the main program
if __name__ == "__main__":
    main()