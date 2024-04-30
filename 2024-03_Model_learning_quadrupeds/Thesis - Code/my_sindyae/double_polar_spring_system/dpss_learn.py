import sys
# Append custom directory to Python's module path
sys.path.append("./my_sindyae/src")
import os
import datetime
import pandas as pd
import numpy as np
# Import TensorFlow with compatibility mode for TensorFlow v1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Import custom functions from other modules within the project
from double_polar_spring_system import get_dpss_data
from sindy_utils import library_size
from training import train_network


### DATA GENERATION ###

# Initialize a time series for data generation
t = np.arange(0, 10, .01)

# Define the coordinate system and simulation settings
coord = 'polar'
real = True
input_type = None
noise_strength = 0

# Generate training and validation datasets for the simulation
training_data = get_dpss_data(100, t, noise_strength=noise_strength, coord=coord, input_type=input_type, real=real)
validation_data = get_dpss_data(25, t, noise_strength=noise_strength, coord=coord, input_type=input_type, real=real)


### PARAMETER SETUP ###

# Flag to control whether to overwrite existing model data
overwrite = True

# Initialize parameter dictionary
params = {}

# Set model configuration parameters
params['pretrained'] = False
params['input_dim'] = 4
params['latent_dim'] = 2

# Set library parameters for the SINDy algorithm
params['library_type'] = 'general'
params['input_type'] = input_type
params['poly_order'] = 1
params['include_goniom'] = True

# Calculate the size of the library based on the configuration
if params['library_type'] == 'general':
    params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_goniom'], params['input_type'], True)
elif params['library_type'] == 'aslip':
    params['library_dim'] = 5

# Define sequential thresholding parameters for SINDy
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.01
params['threshold_frequency'] = 100
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# Set loss weights for different components of the training process
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 0
params['loss_weight_sindy_ddz'] = 1e-4
params['loss_weight_sindy_ddx'] = 1e-6
params['loss_weight_sindy_regularization'] = 1e-9

# Neural network architecture parameters
params['activation'] = 'linear'
params['widths'] = [64,32]

# Set training-specific parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 100
params['learning_rate'] = 1e-3

# BPTT (Back Propagation Through Time) settings for training dynamics models
params['BPTT'] = False
params['prediction_window'] = 5
params['weight_decay_factor'] = 0.9
params['integration_method'] = 'RK2'
params['integration_tstep'] = t[1] - t[0]

# Set paths and logging parameters
params['data_path'] = os.getcwd() + '/my_sindyae/double_polar_spring_system/'
params['print_progress'] = True
params['print_frequency'] = 10

# Set training time constraints
params['max_epochs'] = 1001
params['refinement_epochs'] = 1001


### TRAINING ###

# Define the number of experimental runs
num_experiments = 1
df = pd.DataFrame()

# Loop to perform multiple experiments
for i in range(num_experiments):
    print('EXPERIMENT %d' % (i+1))

    # Check if previous results should be overwritten
    if not overwrite:
        # Create unique identifier for saving results
        params['save_name'] = 'dpss_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    else:
        # Use a generic save name
        params['save_name'] = 'dpss'

    # Reset the TensorFlow default graph for a clean training environment
    tf.reset_default_graph()

    # Train the network with the defined parameters and data, and capture the results
    results_dict = train_network(training_data, validation_data, params)

    # Combine results with parameters and append to dataframe
    df = df.append({**results_dict, **params}, ignore_index=True)

# Conditionally save the training results based on the overwrite flag
if not overwrite:
    df.to_pickle(params['data_path'] + 'dpss_experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')
else:
    df.to_pickle(params['data_path'] + 'dpss_experiment_results.pkl')
