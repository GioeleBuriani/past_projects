import sys
import os
import datetime
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Importing project-specific modules
from polar_spring_system import get_pss_data
from sindy_utils import library_size
from training import train_network

# Setting random seeds for reproducibility
SEED = 0
tf.set_random_seed(SEED)
np.random.seed(SEED)

### DATA GENERATION ###
# Define time series, coordinate system, and noise strength
t = np.arange(0, 20, .02)
coord = 'polar'
real = True
noise_strength = 0

# Generate training and validation data sets
training_data = get_pss_data(100, t, coord=coord, real=real, noise_strength=noise_strength)
validation_data = get_pss_data(25, t, coord=coord, real=real, noise_strength=noise_strength)

### PARAMETER SETUP ###
# Configuration and hyperparameters for the model
overwrite = True  # Flag to determine if previous saves should be overwritten

# Initialize parameters dictionary
params = {}

# Model architecture and function parameters
params['input_dim'] = 2
params['latent_dim'] = 2
params['poly_order'] = 2
params['include_goniom'] = True
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_goniom'], True)

# Sequential thresholding parameters for model refinement
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.1
params['threshold_frequency'] = 50
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# Loss function weighting parameters
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 1e-5
params['loss_weight_sindy_x'] = 1e-5
params['loss_weight_sindy_regularization'] = 1e-9

# Model activation function and layer widths
params['activation'] = 'linear'
params['widths'] = [64, 32]

# Training parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 100
params['learning_rate'] = 1e-3
params['data_path'] = os.getcwd() + '/my_sindyae/polar_spring_system/'
params['print_progress'] = True
params['print_frequency'] = 10

# Training time and epoch configurations
params['max_epochs'] = 101
params['refinement_epochs'] = 51

### TRAINING ###
# Run a series of experiments to train and refine the model
num_experiments = 1
df = pd.DataFrame()

for i in range(num_experiments):
    # Output the experiment iteration
    print('EXPERIMENT %d' % (i+1))

    # Configure saving behavior based on overwrite flag
    if not overwrite:
        params['save_name'] = 'pss_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    else:
        params['save_name'] = 'pss'

    # Reset TensorFlow graph for a fresh start
    tf.reset_default_graph()

    # Train the network and retrieve results
    results_dict = train_network(training_data, validation_data, params)

    # Collect results in a DataFrame
    df = df.append({**results_dict, **params}, ignore_index=True)

# Save experiment results for later analysis
df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')