import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Function to create a heatmap visualization of neural network weights
def create_weights_heatmap(weights, type):
    # Set the default font to Times New Roman for the plot
    mpl.rcParams['font.family'] = 'Times New Roman'

    # Ensure weights are a numpy array, reshape if necessary
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    if weights.ndim == 3 and weights.shape[0] == 1:
        # Reshape from 3D to 2D if the first dimension is 1
        weights = weights.reshape(weights.shape[1], weights.shape[2])
    elif weights.ndim > 2:
        # Raise an error if weights are not 2D or 3D with the first dimension of size 1
        raise ValueError("Weights array must be 2D or 3D with first dimension of size 1")

    # Take the absolute value of weights to focus on magnitude
    weights = np.abs(weights)

    # Define labels for the features, assuming a specific context like robotic joint positions
    feature_labels = [
        'fr.hx Position', 'fr.hy Position', 'fr.kn Position',
        'fl.hx Position', 'fl.hy Position', 'fl.kn Position',
        'rr.hx Position', 'rr.hy Position', 'rr.kn Position',
        'rl.hx Position', 'rl.hy Position', 'rl.kn Position',
        'Body linear Position x', 'Body linear Position y', 'Body linear Position z',
        'Body angular Position x', 'Body angular Position y', 'Body angular Position z'
    ]

    # Customizable font sizes and label paddings for clarity and presentation
    axis_label_font_size = 25
    title_font_size = 25
    tick_label_font_size = 15
    title_pad = 20
    label_pad = 15

    # Initialize the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    if type == 'decoder':
        # Set the plot for decoder weights with appropriate labels and titles
        title = "Heatmap of Decoder Weights"
        ax.set_xticks(np.arange(len(feature_labels)))
        ax.set_xticklabels(feature_labels, fontsize=tick_label_font_size, rotation=90)
        ax.set_yticks(np.arange(weights.shape[0]))
        ax.set_yticklabels([f"z_{i+1}" for i in range(weights.shape[0])], fontsize=tick_label_font_size)
    else:
        # Set the plot for encoder weights with appropriate labels and titles
        title = "Heatmap of the Transpose of the Encoder's Weight Matrix"
        ax.set_xticks(np.arange(weights.shape[1]))
        ax.set_xticklabels([f"z_{i+1}" for i in range(weights.shape[1])], fontsize=tick_label_font_size)
        ax.set_yticks(np.arange(len(feature_labels)))
        ax.set_yticklabels(feature_labels, fontsize=tick_label_font_size)

    # Adjust label padding for the x and y axis
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad

    # Adjust tick parameters for size
    ax.tick_params(axis='x', labelsize=tick_label_font_size, rotation=90)
    ax.tick_params(axis='y', labelsize=tick_label_font_size)

    # Create an image with the weights matrix using a colormap
    im = ax.imshow(weights, cmap='hot', aspect='auto')
    # Add a color bar to indicate the scale of weights
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=tick_label_font_size)  # Adjust colorbar tick label size

    # Set the title for the heatmap
    plt.title(title, fontsize=title_font_size, pad=title_pad)
    # Adjust layout to fit everything neatly
    plt.tight_layout()
    plt.show(block=False)

# Function to plot joint data and compare it with learned data from simulations
def plot_joint_data_3(time, joint_data, learned_joint_data_1=None, learned_joint_data_2=None, save_filename=None):
    # Set the font properties globally for all plots to Times New Roman and specify other aesthetics
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20  # Set the base font size to 20 for better visibility
    mpl.rcParams['axes.titlesize'] = 25  # Title size for subplots
    mpl.rcParams['axes.labelsize'] = 25  # Label size for axes titles
    mpl.rcParams['xtick.labelsize'] = 15  # Font size for x-axis tick labels
    mpl.rcParams['ytick.labelsize'] = 15  # Font size for y-axis tick labels
    mpl.rcParams['legend.fontsize'] = 20  # Font size for legends
    mpl.rcParams['lines.linewidth'] = 3  # Width of the plot lines for better visibility

    # Scale the joint data for visualization
    joint_data = joint_data * 10
    if learned_joint_data_1 is not None:
        learned_joint_data_1 = learned_joint_data_1 * 10
    if learned_joint_data_2 is not None:
        learned_joint_data_2 = learned_joint_data_2 * 10

    # Determine the global minimum and maximum values across all datasets to set uniform y-axis limits
    global_min = np.min(joint_data)
    global_max = np.max(joint_data)
    if learned_joint_data_1 is not None:
        global_min = min(global_min, np.min(learned_joint_data_1))
        global_max = max(global_max, np.max(learned_joint_data_1))
    if learned_joint_data_2 is not None:
        global_min = min(global_min, np.min(learned_joint_data_2))
        global_max = max(global_max, np.max(learned_joint_data_2))

    # Set margins for the y-axis to prevent any data from being on the edge of the plot
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a figure and axes for the subplots arranged in a 3x4 grid
    fig, axs = plt.subplots(4, 3, figsize=(19.2, 20.16))  # Specify a large figure size for clarity
    axs = axs.flatten()  # Flatten the 2D array of axes to facilitate iterating over them

    # Define names for each joint to use as titles in the subplots
    plot_names = [
        'fr.hx Position', 'fr.hy Position', 'fr.kn Position',
        'fl.hx Position', 'fl.hy Position', 'fl.kn Position',
        'rr.hx Position', 'rr.hy Position', 'rr.kn Position',
        'rl.hx Position', 'rl.hy Position', 'rl.kn Position'
    ]

    # Iterate over each joint to plot the real and simulated data
    for joint_idx in range(joint_data.shape[1]):
        axs[joint_idx].plot(time, joint_data[:, joint_idx], label='Real data')
        if learned_joint_data_1 is not None:
            axs[joint_idx].plot(time, learned_joint_data_1[:, joint_idx], label='Simulated data', color='orange')
        if learned_joint_data_2 is not None:
            axs[joint_idx].plot(time, learned_joint_data_2[:, joint_idx], label='Simulated data 0.05s', color='green')

        # Set title for each subplot using the defined names
        axs[joint_idx].set_title(plot_names[joint_idx])

        # Label x and y axes for only the outer plots to avoid clutter
        if joint_idx >= 9:  # Label x-axis for the bottom row
            axs[joint_idx].set_xlabel('Time (s)')
        if joint_idx % 3 == 0:  # Label y-axis for the first column
            axs[joint_idx].set_ylabel('Angle (rad)')

        # Set the same y-axis limits for all subplots for consistency
        axs[joint_idx].set_ylim(y_min, y_max)
        axs[joint_idx].grid(True)  # Enable grid for better readability
        axs[joint_idx].legend()  # Show legend in each subplot

    # Adjust the layout to prevent label overlap
    plt.tight_layout()

    # Save the figure if a filename is provided, otherwise display it
    if save_filename:
        plt.savefig(save_filename, dpi=100)  # Save the figure with high resolution
        plt.show(block=False)
    else:
        plt.show(block=False)  # Display the plot without blocking the rest of the script

# Function to plot body data alongside learned data from simulations, with the option to save the figure
def plot_body_data_3(time, body_data, learned_body_data_1=None, learned_body_data_2=None, save_filename=None):
    # Configure plot aesthetics using Matplotlib's settings
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20  # Set general font size
    mpl.rcParams['axes.titlesize'] = 25  # Title size for subplots
    mpl.rcParams['axes.labelsize'] = 25  # Axis label size
    mpl.rcParams['xtick.labelsize'] = 15  # X-axis tick label size
    mpl.rcParams['ytick.labelsize'] = 15  # Y-axis tick label size
    mpl.rcParams['legend.fontsize'] = 20  # Legend font size
    mpl.rcParams['lines.linewidth'] = 3   # Width of the plot lines

    # Scale the body data for visualization purposes
    body_data = body_data * 10
    if learned_body_data_1 is not None:
        learned_body_data_1 = learned_body_data_1 * 10
    if learned_body_data_2 is not None:
        learned_body_data_2 = learned_body_data_2 * 10

    # Determine global minimum and maximum values for setting consistent y-axis limits
    global_min = np.min(body_data)
    global_max = np.max(body_data)

    # Update the global minimum and maximum based on the learned data
    if learned_body_data_1 is not None:
        global_min = min(global_min, np.min(learned_body_data_1))
        global_max = max(global_max, np.max(learned_body_data_1))
    
    if learned_body_data_2 is not None:
        global_min = min(global_min, np.min(learned_body_data_2))
        global_max = max(global_max, np.max(learned_body_data_2))

    # Establish margins for the y-axis limits to ensure data is clearly visible and not on the edge
    margin = 0.1 * (global_max - global_min)
    y_min_lin = 0 - margin  # For linear position plots (first row)
    y_min_ang = global_min - margin  # For angular position plots (second row)
    y_max = global_max + margin

    # Create a figure with multiple subplots arranged in a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(19.2, 10.8))
    axs = axs.flatten()  # Flatten the array of axes for easy indexing

    # Define names for the plot titles based on the data being visualized
    plot_names = ['Body linear Position x', 'Body linear Position y', 'Body linear Position z',
                  'Body angular Position x', 'Body angular Position y', 'Body angular Position z']

    # Loop through each subplot to plot the real and simulated data
    for i in range(6):
        axs[i].plot(time, body_data[:, i], label='Real data')
        if learned_body_data_1 is not None:
            axs[i].plot(time, learned_body_data_1[:, i], label='Simulated data full', color='orange')
        if learned_body_data_2 is not None:
            axs[i].plot(time, learned_body_data_2[:, i], label='Simulated data 0.05s', color='green')

        # Set titles and axis labels selectively to avoid clutter
        axs[i].set_title(plot_names[i])
        if i >= 3:  # Set x-axis labels for the bottom row
            axs[i].set_xlabel('Time (s)')
        if i == 0 or i == 3:  # Set y-axis labels for the first column
            axs[i].set_ylabel('Position (m)' if i < 3 else 'Orientation (rad)')

        # Set y-axis limits based on whether the plot is for linear or angular data
        axs[i].set_ylim(y_min_lin if i < 3 else y_min_ang, y_max)
        axs[i].grid(True)  # Enable grid for better readability
        axs[i].legend()

    # Adjust layout to prevent overlap of plot elements
    plt.tight_layout()

    # Save the figure if a filename is provided; otherwise, display it
    if save_filename:
        plt.savefig(save_filename, dpi=100)  # Save the figure with high resolution
        plt.show(block=False)
    else:
        plt.show(block=False)  # Display the plot without blocking script execution

# Function to plot latent data alongside learned latent data for visualization
def plot_latent_data_2(time, latent_data, learned_latent_data_1=None, learned_latent_data_2=None, save_filename=None):
    # Configure plot aesthetics using Matplotlib's settings
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20  # Set general font size for better readability
    mpl.rcParams['axes.titlesize'] = 25  # Font size for subplot titles
    mpl.rcParams['axes.labelsize'] = 25  # Font size for axis labels
    mpl.rcParams['xtick.labelsize'] = 15  # Font size for x-axis tick labels
    mpl.rcParams['ytick.labelsize'] = 15  # Font size for y-axis tick labels
    mpl.rcParams['legend.fontsize'] = 20  # Font size for legend
    mpl.rcParams['lines.linewidth'] = 3  # Thickness of the plot lines

    # Scale the latent data by 10 for better visibility in plots
    latent_data = latent_data * 10
    if learned_latent_data_1 is not None:
        learned_latent_data_1 = learned_latent_data_1 * 10
    if learned_latent_data_2 is not None:
        learned_latent_data_2 = learned_latent_data_2 * 10

    # Calculate the global minimum and maximum values to set consistent y-axis limits across plots
    global_min = np.min(latent_data)
    global_max = np.max(latent_data)

    # Update the global minimum and maximum based on learned data sets
    if learned_latent_data_1 is not None:
        global_min = min(global_min, np.min(learned_latent_data_1))
        global_max = max(global_max, np.max(learned_latent_data_1))
    
    if learned_latent_data_2 is not None:
        global_min = min(global_min, np.min(learned_latent_data_2))
        global_max = max(global_max, np.max(learned_latent_data_2))

    # Set a margin for the y-axis to ensure data is not plotted on the edge
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 10.8))  # Set figure size
    axs = axs.flatten()  # Flatten the array of axes for easy indexing

    # Define titles for each subplot, assuming specific context of the latent variables
    plot_names = ['Latent Acceleration 1', 'Latent Acceleration 2']

    # Plot the latent data and any learned models' predictions
    for i in range(2):
        axs[i].plot(time, latent_data[:, i], label='Latent data')
        if learned_latent_data_1 is not None:
            axs[i].plot(time, learned_latent_data_1[:, i], label='Simulated latent data 0.1s', linestyle='--')
        if learned_latent_data_2 is not None:
            axs[i].plot(time, learned_latent_data_2[:, i], label='Simulated data 0.1s', linestyle=':')

        # Set the title for each subplot based on predefined names
        axs[i].set_title(plot_names[i])

        # Set x-axis labels (only needed on the bottom if multiple rows)
        axs[i].set_xlabel('Time (s)')

        # Set consistent y-axis limits across subplots for clarity
        axs[i].set_ylim(y_min, y_max)
        axs[i].grid(True)  # Enable grid for easier data viewing
        axs[i].legend()  # Show legend to identify real vs. simulated data

    # Adjust the layout to prevent overlap of elements
    plt.tight_layout()

    # Save the figure to a file if a filename is provided, otherwise display it
    if save_filename:
        plt.savefig(save_filename, dpi=100)  # Save the figure with specified DPI for high quality
        plt.show(block=False)
    else:
        plt.show(block=False)  # Display the plot without blocking the execution of subsequent code

# Function to plot four dimensions of latent data and compare with up to two sets of learned data
def plot_latent_data_4(time, latent_data, learned_latent_data_1=None, learned_latent_data_2=None, save_filename=None):
    # Configure the visual appearance of plots globally using Matplotlib parameters
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20  # Increase the base font size for better readability
    mpl.rcParams['axes.titlesize'] = 25  # Font size for the titles of subplots
    mpl.rcParams['axes.labelsize'] = 25  # Font size for the x and y axis labels of plots
    mpl.rcParams['xtick.labelsize'] = 15  # Font size for the x-axis tick labels
    mpl.rcParams['ytick.labelsize'] = 15  # Font size for the y-axis tick labels
    mpl.rcParams['legend.fontsize'] = 20  # Font size for legends
    mpl.rcParams['lines.linewidth'] = 3   # Width of the lines in plots

    # Scale latent data for better visualization
    latent_data = latent_data * 10
    if learned_latent_data_1 is not None:
        learned_latent_data_1 = learned_latent_data_1 * 10
    if learned_latent_data_2 is not None:
        learned_latent_data_2 = learned_latent_data_2 * 10

    # Calculate the global minimum and maximum across the latent data to set consistent plot limits
    global_min = np.min(latent_data)
    global_max = np.max(latent_data)

    # Adjust the global minimum and maximum based on learned data if available
    if learned_latent_data_1 is not None:
        global_min = min(global_min, np.min(learned_latent_data_1))
        global_max = max(global_max, np.max(learned_latent_data_1))
    
    if learned_latent_data_2 is not None:
        global_min = min(global_min, np.min(learned_latent_data_2))
        global_max = max(global_max, np.max(learned_latent_data_2))

    # Define a margin to ensure data does not sit on the plot edges
    margin = 0.1 * (global_max - global_min)
    y_min = global_min - margin
    y_max = global_max + margin

    # Setup a 2x2 subplot grid for visualizing four dimensions of latent data
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 10.8))  # Define the figure size
    axs = axs.flatten()  # Flatten the array of axes for simpler access

    # Define titles for each of the four subplots corresponding to the latent dimensions
    plot_names = ['Latent Acceleration 1', 'Latent Acceleration 2', 'Latent Acceleration 3', 'Latent Acceleration 4']

    # Plot each dimension of latent data alongside the corresponding learned data
    for i in range(4):
        axs[i].plot(time, latent_data[:, i], label='Latent data')
        if learned_latent_data_1 is not None:
            axs[i].plot(time, learned_latent_data_1[:, i], label='Simulated latent data full', color='orange')
        if learned_latent_data_2 is not None:
            axs[i].plot(time, learned_latent_data_2[:, i], label='Simulated latent data 0.1s', color='green')

        # Set the title for each subplot based on the defined names
        axs[i].set_title(plot_names[i])

        # Only set x-axis labels for the bottom plots to avoid clutter
        if i >= 2:
            axs[i].set_xlabel('Time (s)')

        # Set consistent y-axis limits based on previously determined minimum and maximum
        axs[i].set_ylim(y_min, y_max)
        axs[i].grid(True)  # Enable grid for better visibility of data points
        axs[i].legend()  # Display legend to identify data and model predictions

    # Automatically adjust subplot layouts to prevent overlap
    plt.tight_layout()

    # Save the figure if a filename is provided; otherwise, display it
    if save_filename:
        plt.savefig(save_filename, dpi=100)  # Save the figure with high resolution
        plt.show(block=False)
    else:
        plt.show(block=False)  # Display the figure without blocking script execution

# Function to plot the impact of latent dimensionality on model performance metrics
def dimensionality_plots(n_dims, coefficients_data, decoder_losses_data, lambda_value=1e-3):
    # Set the default font for all plot elements to Times New Roman
    mpl.rcParams['font.family'] = 'Times New Roman'

    # Generate an array of latent dimensions from 1 to n_dims
    latent_dims = np.arange(1, n_dims + 1)
    
    # Calculate the mean and standard deviation for coefficients and decoder losses across different runs
    coefficients_mean = coefficients_data.mean(axis=1)
    coefficients_std = coefficients_data.std(axis=1)
    decoder_losses_mean = decoder_losses_data.mean(axis=1)
    decoder_losses_std = decoder_losses_data.std(axis=1)
    
    # Calculate combined loss which might be used for model selection or hyperparameter tuning
    combined_loss = 2*decoder_losses_mean + 2*lambda_value * np.log(coefficients_mean)

    # Define custom font sizes and other plot settings for clarity and presentation
    axis_label_font_size = 25
    title_font_size = 25
    tick_label_font_size = 15
    legend_font_size = 20
    line_width = 3
    marker_size = 10
    title_pad = 20
    label_pad = 15

    # Plot 1: Decoder Loss vs. Number of Latent Dimensions
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()  # Get the current axis for plotting decoder loss
    # Plot decoder loss with error bars to represent variability
    ax1.errorbar(latent_dims, decoder_losses_mean, yerr=decoder_losses_std, fmt='-o', color='b', ecolor='blue', elinewidth=2, capsize=4, label='Decoder Error', linewidth=line_width, markersize=marker_size)
    ax1.set_xlabel('Number of Latent Dimensions', fontsize=axis_label_font_size, labelpad=label_pad)
    ax1.set_ylabel('Decoder Error', color='b', fontsize=axis_label_font_size, labelpad=label_pad)
    ax1.tick_params(axis='y', labelcolor='b', labelsize=tick_label_font_size)
    ax1.tick_params(axis='x', labelsize=tick_label_font_size)
    ax1.set_xticks(latent_dims)
    ax1.grid(True)

    # Add a second y-axis for the number of coefficients
    ax2 = ax1.twinx()  # Create a twin y-axis sharing the same x-axis
    ax2.errorbar(latent_dims, coefficients_mean, yerr=coefficients_std, fmt='-s', color='g', ecolor='green', elinewidth=2, capsize=4, label='Number of Coefficients', linewidth=line_width, markersize=marker_size)
    ax2.set_ylabel('Number of Coefficients', color='g', fontsize=axis_label_font_size, labelpad=label_pad)
    ax2.tick_params(axis='y', labelcolor='g', labelsize=tick_label_font_size)
    
    plt.title('Decoder Error and Number of Coefficients per Dimensionality of Latent Space', fontsize=title_font_size, pad=title_pad)
    ax1.legend(loc='upper left', fontsize=legend_font_size)
    ax2.legend(loc='upper right', fontsize=legend_font_size)
    plt.show()
    
    # Plot 2: Combined Loss vs. Number of Latent Dimensions
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims, combined_loss, marker='o', color='r', linewidth=line_width, markersize=marker_size)
    plt.title('Combined Loss per Dimensionality of Latent Space', fontsize=title_font_size, pad=title_pad)
    plt.xlabel('Number of Latent Dimensions', fontsize=axis_label_font_size, labelpad=label_pad)
    plt.ylabel('Combined Loss', fontsize=axis_label_font_size, labelpad=label_pad)
    plt.xticks(latent_dims, fontsize=tick_label_font_size)
    plt.yticks(fontsize=tick_label_font_size)
    plt.grid(True)
    plt.show()

# Main function to perform analysis of model performance across different latent dimensions
def main():
    # Specify the number of dimensions to evaluate
    n_dims = 18

    # Coefficients data simulating different scenarios over several runs
    coefficients = np.array([
        [6, 6, 7, 5, 6],
        [19, 16, 20, 18, 24],
        [27, 27, 26, 28, 28],
        [42, 36, 43, 42, 41],
        [58, 58, 62, 52, 56],
        [88, 89, 79, 95, 97],
        [108, 125, 112, 150, 126],
        [140, 139, 149, 150, 126],
        [160, 164, 168, 170, 138],
        [232, 216, 213, 170, 252],
        [309, 337, 364, 291, 261],
        [334, 337, 364, 390, 315],
        [400, 426, 480, 376, 433],
        [433, 486, 375, 498, 413],
        [578, 638, 544, 605, 753],
        [872, 747, 750, 974, 1043],
        [981, 1161, 1207, 814, 955],
        [1321, 1233, 1007, 1626, 1425]
    ])
    
    # Decoder losses data simulating variability in model decoding performance across several runs
    decoder_losses = np.array([
        [0.01318, 0.0144506, 0.01336382, 0.01225457, 0.01298958],
        [0.011531, 0.01052548, 0.01253488, 0.01052297, 0.01054065],
        [0.007373, 0.00836133, 0.00937126, 0.00737202, 0.00832757],
        [0.007085, 0.00745792, 0.00657256, 0.00627265, 0.00695927],
        [0.006870, 0.00617426, 0.00648202, 0.0070679, 0.00636407],
        [0.006072, 0.00639572, 0.00665128, 0.00574657, 0.00586591],
        [0.005956, 0.00645396, 0.00625865, 0.00644937, 0.00574501],
        [0.005892, 0.00621049, 0.0057906, 0.00569281, 0.00558997],
        [0.005887, 0.00638206, 0.0056966, 0.00589061, 0.00608425],
        [0.005892, 0.00619437, 0.0059209, 0.00559088, 0.00571245],
        [0.005667, 0.00556647, 0.00596595, 0.00537243, 0.00586397],
        [0.009572, 0.00570713, 0.00568122, 0.00607683, 0.00545431],
        [0.005501, 0.00540725, 0.00577509, 0.00520052, 0.00581439],
        [0.002849, 0.00580442, 0.00560065, 0.00575079, 0.00529219],
        [0.002948, 0.00520338, 0.00560429, 0.00580731, 0.00559847],
        [0.005198, 0.00480629, 0.00545425, 0.0052282, 0.00508939],
        [0.005294, 0.00539371, 0.00522119, 0.00559041, 0.00511429],
        [0.005182, 0.00509263, 0.00538935, 0.00511242, 0.00498466]
    ])

    # Execute the plotting function that evaluates the impact of latent dimensionality on model performance
    dimensionality_plots(n_dims, coefficients, decoder_losses)

# Execute the main function when the script is run as the main program
if __name__ == "__main__":
    main()