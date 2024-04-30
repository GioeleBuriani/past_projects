import sys
# Add a custom path to the system path for importing modules
sys.path.append("./my_sindyae/src")
import os
import pickle
import tensorflow.compat.v1 as tf
# Disable TensorFlow v2 behavior as the script is based on v1 API
tf.disable_v2_behavior()
import numpy as np

import matplotlib.pyplot as plt

# Import modules and functions from other scripts in the project
from double_polar_spring_system import generate_dpss_data, plot_dpss, plot_dpss_compar
from autoencoder import full_network, decode_latent_representation
from sindy_utils import sindy_batch_simulate_order2
from training import create_feed_dictionary

# Set the coordinate system to be used in the simulation
coord = 'polar'
input_type = None
real = True

# Get the current working directory and construct the data path for parameter files
data_path = os.getcwd() + '/my_sindyae/double_polar_spring_system/'
# Define the save name for the model to match with saved parameters
save_name = 'dpss'
# Load the parameters from a pickle file
params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))

# Create the autoencoder network using the loaded parameters
autoencoder_network = full_network(params)
# Create a placeholder for the learning rate in TensorFlow
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
# Create a saver object to save and restore variables in the TensorFlow graph
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

# Create an empty tuple to store TensorFlow operations or tensors for running
tensorflow_run_tuple = ()
# Append each TensorFlow operation or tensor in the autoencoder network to the tuple
for key in autoencoder_network.keys():
    tensorflow_run_tuple += (autoencoder_network[key],)

# Set the number of time steps to simulate
t_test = np.arange(0, 15, .01)

# Set the initial state of the system using random values
x0_test = np.array([np.random.uniform(- np.pi / 2, np.pi / 2), np.random.uniform(0.5, 2.5), np.random.uniform(- np.pi / 2, np.pi / 2), np.random.uniform(0.5, 2.5)])
dx0_test = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
state0_test = np.array([[x0_test[0], x0_test[1], x0_test[2], x0_test[3], dx0_test[0], dx0_test[1], dx0_test[2], dx0_test[3]]])

# Generate the test data using the double polar spring system module
data_test = generate_dpss_data(state0_test, t_test, normalization=np.array([1/10, 1/10, 1/10, 1/10]), coord=coord, real=real, input_type=input_type)
# Reshape the data to fit the network input dimensions
data_test['x'] = data_test['x'].reshape((-1, 4))
data_test['dx'] = data_test['dx'].reshape((-1, 4))
data_test['ddx'] = data_test['ddx'].reshape((-1, 4))
data_test['u'] = data_test['u'].reshape((-1, 4))

# Create and manage a TensorFlow session
with tf.Session() as sess:
    # Initialize all TensorFlow variables with their default values in the session
    sess.run(tf.global_variables_initializer())
    # Restore the saved model using the saver object and the specified path and save_name
    saver.restore(sess, data_path + save_name)
    # Create a feed dictionary for the test data using the create_feed_dictionary function
    test_dictionary = create_feed_dictionary(data_test, params)
    # Run the TensorFlow operations or tensors specified in the tensorflow_run_tuple on the test data
    tf_results = sess.run(tensorflow_run_tuple, feed_dict=test_dictionary)

    # Create an empty dictionary to store the results for the test set
    test_set_results = {}
    # Assign each result to the corresponding key in the test_set_results dictionary
    for i, key in enumerate(autoencoder_network.keys()):
        test_set_results[key] = tf_results[i]

    # Simulate the system dynamics using SINDy, applied on latent space representations
    z_sim, dz_sim, ddz_sim = sindy_batch_simulate_order2(test_set_results['z'],
                                                         test_set_results['dz'],
                                                         test_set_results['u_latent'],
                                                         t_test,
                                                         params['coefficient_mask'] * test_set_results['sindy_coefficients'],
                                                         params['poly_order'],
                                                         params['include_goniom'],
                                                         input_type,
                                                         batch_size=100
                                                         )

    # Decode the latent representations back to the original space
    x_sim = decode_latent_representation(z_sim, autoencoder_network['decoder_weights'], autoencoder_network['decoder_biases'], params['activation'])
    dx_sim = decode_latent_representation(dz_sim, autoencoder_network['decoder_weights'], autoencoder_network['decoder_biases'], params['activation'])
    ddx_sim = decode_latent_representation(ddz_sim, autoencoder_network['decoder_weights'], autoencoder_network['decoder_biases'], params['activation'])
    # Evaluate the decoded latent representation (convert from TensorFlow tensor to numpy array)
    x_sim_eval = sess.run(x_sim)
    dx_sim_eval = sess.run(dx_sim)
    ddx_sim_eval = sess.run(ddx_sim)


# Plot comparison of real and simulated data using the matplotlib library
# plot_dpss(t_test, data_test['x'], name='Real data', coord=coord, title='Positions Real Space')
# plot_dpss(t_test, test_set_results['z'], name='Latent data', coord=coord, title='Positions Latent Space')
# plot_dpss(t_test, test_set_results['dz'], name='Latent data', coord=coord, title='Velocities Latent Space')
# plot_dpss(t_test, test_set_results['ddz'], name='Latent data', coord=coord, title='Accelerations Latent Space')  
plot_dpss_compar(t_test, data_test['x'], x_sim_eval, name_1='Real data', name_2='Simulated real data', coord=coord, title='Positions Real Space')
# plot_dpss_compar(t_test, test_set_results['z'], z_sim, name_1='Latent data', name_2='Simulated latent data', coord=coord, title='Positions Latent Space')
# plot_dpss_compar(t_test, data_test['dx'], dx_sim_eval, name_1='Real data', name_2='Simulated real data', coord=coord, title='Velocities Real Space')
# plot_dpss_compar(t_test, test_set_results['dz'], dz_sim, name_1='Latent data', name_2='Simulated latent data', coord=coord, title='Velocities Latent Space')
# plot_dpss_compar(t_test, data_test['ddx'], ddx_sim_eval, name_1='Real data', name_2='Simulated real data', coord=coord, title='Accelerations Real Space')
# plot_dpss_compar(t_test, test_set_results['ddz'], ddz_sim, name_1='Latent data', name_2='Simulated latent data', coord=coord, title='Accelerations Latent Space')   

plt.show()