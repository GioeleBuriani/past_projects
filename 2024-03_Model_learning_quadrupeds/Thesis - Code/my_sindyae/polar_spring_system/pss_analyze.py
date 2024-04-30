import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

# Disable eager execution to use TensorFlow 1.x functionalities
tf.disable_v2_behavior()

# Importing necessary modules from the project
from polar_spring_system import generate_pss_data, plot_pss_compar
from autoencoder import full_network, decode_latent_representation
from sindy_utils import sindy_simulate_order2
from training import create_feed_dictionary

# Setting up coordinate system and data path for model loading
coord = 'polar'
real = False
data_path = os.getcwd() + '/my_sindyae/polar_spring_system/'
save_name = 'pss'

# Load model parameters and initialize the autoencoder
params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))
autoencoder_network = full_network(params)

# Initialize TensorFlow placeholders and saver for session management
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

# Bundle all autoencoder operations for batch execution
tensorflow_run_tuple = tuple(autoencoder_network.values())

# Define simulation parameters for generating test data
t_test = np.arange(0, 30, .02)
x0_test = np.array([np.random.uniform(- np.pi / 2, np.pi / 2), np.random.uniform(0.5, 2.5)])
dx0_test = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
state0_test = np.array([x0_test.tolist() + dx0_test.tolist()])

# Generate and preprocess test data
data_test = generate_pss_data(state0_test, t_test, normalization=np.array([1/10, 1/10]), coord=coord, real=real)
data_test['x'] = data_test['x'].reshape((-1, 2))
data_test['dx'] = data_test['dx'].reshape((-1, 2))
data_test['ddx'] = data_test['ddx'].reshape((-1, 2))

# Start TensorFlow session for executing the model
with tf.Session() as sess:
    # Initialize all global variables in the session
    sess.run(tf.global_variables_initializer())

    # Restore the saved model using the constructed data path
    saver.restore(sess, data_path + save_name)

    # Prepare test data for feeding into the model
    test_dictionary = create_feed_dictionary(data_test, params)

    # Execute the model to obtain results
    tf_results = sess.run(tensorflow_run_tuple, feed_dict=test_dictionary)

    # Mapping TensorFlow outputs to corresponding network components for analysis
    test_set_results = {key: value for key, value in zip(autoencoder_network.keys(), tf_results)}

    # Simulate the system's future behavior using SINDy
    z_sim, dz_sim = sindy_simulate_order2(test_set_results['z'][0],
                                          test_set_results['dz'][0],
                                          t_test,
                                          params['coefficient_mask'] * test_set_results['sindy_coefficients'],
                                          params['poly_order'],
                                          params['include_goniom'])

    # Decode latent space representation to original state space
    x_sim = decode_latent_representation(z_sim, autoencoder_network['decoder_weights'], autoencoder_network['decoder_biases'], params['activation'])
    x_sim_eval = sess.run(x_sim)

# Plotting the comparison between actual data and SINDy simulated data
plot_pss_compar(t_test, test_set_results['z'], z_sim, name_1='Latent data', name_2='Simulated latent data', coord=coord, title='Latent Space Comparison')
plot_pss_compar(t_test, data_test['x'], x_sim_eval, name_1='Real data', name_2='Simulated real data', coord=coord, title='Real Space Comparison')