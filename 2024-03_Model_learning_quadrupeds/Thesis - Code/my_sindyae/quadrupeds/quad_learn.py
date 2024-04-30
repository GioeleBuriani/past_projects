import sys
sys.path.append("./my_sindyae/src")  # Add the directory containing custom modules to Python's search path
import os
import datetime
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow v2 behavior as the code uses v1 features

from go1_data import create_go1_dataset
from a1_data import create_a1_dataset
from sindy_utils import library_size
from training import train_network

# Define filenames for training and validation datasets
training_file = 'a1_sim_jump_dataset_training'
validation_file = 'a1_sim_jump_dataset_validation'

# Construct full file paths to the datasets
training_path = 'data/a1_sim/' + training_file + '.csv'
validation_path = 'data/a1_sim/' + validation_file + '.csv'

### DATA ACQUISITION ###
# Load and preprocess training and validation data
training_data = create_a1_dataset(training_path, normalization=1/10, apply_filter=False)
validation_data = create_a1_dataset(validation_path, normalization=1/10, apply_filter=False)

### PARAMETER SETUP ###
# Initialize parameters for training and model configuration
overwrite = True  # Flag to control whether to overwrite existing results
save_name = ''  # Base name for saved files

params = {}

# NOTE: If pretrained is set to True, specify the model name in src/autoencoder.py line 227 and the trainable variables
# in src/training.py line 31
params['pretrained'] = False  # Start training without pre-trained weights

params['input_dim'] = 18  # Dimensionality of input data
params['latent_dim'] = 4  # Dimensionality of the latent space

# Configuration for the library of differential equations used in the SINDy algorithm
params['library_type'] = 'general'
params['input_type'] = 'real'
params['poly_order'] = 1
params['include_goniom'] = True

# Calculate library dimensions based on configuration
if params['library_type'] == 'general':
    params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_goniom'], params['input_type'], True)
elif params['library_type'] == 'aslip':
    params['library_dim'] = 5

# Setup for sequential thresholding in the SINDy algorithm
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.01
params['threshold_frequency'] = 100
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

# NOTE: If coefficient_initialization is set to 'pretrained', specify the model name in src/autoencoder.py line 70
params['coefficient_initialization'] = 'constant'  # Initialize coefficients to a constant value
# params['coefficient_initialization'] = 'pretrained'  # Initialize coefficients with pre-trained values

# Weighting of different components in the loss function
params['loss_weight_decoder'] = 1
params['loss_weight_sindy_z'] = 0
params['loss_weight_sindy_ddz'] = 1e-4
params['loss_weight_sindy_ddx'] = 1e-6
params['loss_weight_sindy_regularization'] = 1e-6

params['activation'] = 'linear'  # Activation function for the neural network layers
params['widths'] = [64, 32]  # Widths of the layers in the neural network

# Training configuration parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 100
params['learning_rate'] = 1e-3

# Backpropagation through time (BPTT) parameters
params['BPTT'] = False
params['prediction_window'] = 5
params['weight_decay_factor'] = 0.9
params['integration_method'] = 'RK2'
params['integration_tstep'] = 0.01

params['data_path'] = os.getcwd() + '/my_sindyae/quadrupeds/Results/'
params['print_progress'] = True
params['print_frequency'] = 10

# Set the number of epochs for training and refinement phases
params['max_epochs'] = 501
params['refinement_epochs'] = 101

### TRAINING ###
num_experiments = 1  # Number of training experiments to perform
df = pd.DataFrame()  # Initialize a DataFrame to store results

# Run multiple experiments and gather results
for i in range(num_experiments):
    print('EXPERIMENT %d' % (i + 1))
    if not overwrite:
        # Append a timestamp to the save name if not overwriting
        params['save_name'] = 'a1_' + save_name
    else:
        # Use a fixed save name if overwriting
        params['save_name'] = 'a1'

    tf.reset_default_graph()  # Reset TensorFlow graph to clear previous settings

    # Train the model and capture results
    results_dict = train_network(training_data, validation_data, params)

    # Add results and parameters to the DataFrame
    df = df.append({**results_dict, **params}, ignore_index=True)

# Save the experiment results to a pickle file
if not overwrite:
    df.to_pickle(params['data_path'] + 'a1_' + save_name + '_experiment_results.pkl')
else:
    df.to_pickle(params['data_path'] + 'a11_experiment_results.pkl')