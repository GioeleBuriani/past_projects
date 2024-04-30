import sys
import os
import pickle
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

# Add custom modules to Python's search path
sys.path.append("./my_sindyae/src")

# Disable TensorFlow v2 behavior to ensure compatibility with v1 code
tf.disable_v2_behavior()

from go1_data import create_go1_dataset, plot_joint_data, plot_body_data, plot_torque_data, plot_forces_on_com, plot_go1_compar
from a1_data import create_a1_dataset
from autoencoder import full_network, decode_latent_representation
from sindy_utils import sindy_batch_simulate_order2, hybrid_sindy_batch_simulate_order2, write_learnt_equations
from training import create_feed_dictionary
from go1_plots import create_weights_heatmap, plot_body_data_3, plot_joint_data_3, plot_latent_data_2, plot_latent_data_4
from aslip import hybrid_aslip_batch_simulate, plot_aslip, plot_aslip_3

# Flag to determine whether to use hybrid SINDy model
hybrid = False

# Define save names and file for testing
save_name = 'a1'
save_name_2 = ''
test_file = 'a1_sim_jump_dataset_test'
datapoints_per_jump = 1200
jump_n = 1

# Construct the path to the test dataset
test_path = 'data/a1_sim/' + test_file + '.csv'

# Load and preprocess the test dataset
test_data = create_a1_dataset(test_path, normalization=1/10, apply_filter=False)

# Select data corresponding to a specific jump
test_data['t'] = test_data['t'][(jump_n-1)*datapoints_per_jump:jump_n*datapoints_per_jump]
test_data['x'] = test_data['x'][(jump_n-1)*datapoints_per_jump:jump_n*datapoints_per_jump, :]
test_data['dx'] = test_data['dx'][(jump_n-1)*datapoints_per_jump:jump_n*datapoints_per_jump, :]
test_data['ddx'] = test_data['ddx'][(jump_n-1)*datapoints_per_jump:jump_n*datapoints_per_jump, :]
test_data['u'] = test_data['u'][(jump_n-1)*datapoints_per_jump:jump_n*datapoints_per_jump, :]
test_data['foot'] = test_data['foot'][(jump_n-1)*datapoints_per_jump:jump_n*datapoints_per_jump, :]

# Establish the directory for saving and loading model data
data_path = os.getcwd() + '/my_sindyae/quadrupeds/Results/'

# Load model parameters and results from pickle files
params = pickle.load(open(data_path + save_name + '_params.pkl', 'rb'))
results = pickle.load(open(data_path + save_name + '_experiment_results.pkl', 'rb'))

# Initialize and build the autoencoder network
autoencoder_network = full_network(params)
# Setup TensorFlow placeholders and saver for managing the network
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

# Tuple to store TensorFlow operations for execution
tensorflow_run_tuple = ()
# Populate tuple with operations from the autoencoder network
for key in autoencoder_network.keys():
    tensorflow_run_tuple += (autoencoder_network[key],)

# TensorFlow session for executing the model
with tf.Session() as sess:
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    # Load the trained model
    saver.restore(sess, data_path + save_name)
    # Prepare data for model input
    test_dictionary = create_feed_dictionary(test_data, params)
    # Execute model operations
    tf_results = sess.run(tensorflow_run_tuple, feed_dict=test_dictionary)

    # Store results from the session
    test_set_results = {}
    for i, key in enumerate(autoencoder_network.keys()):
        test_set_results[key] = tf_results[i]

    # Simulate the dynamics using the SINDy model if the hybrid flag is not set
    if not hybrid:
        # Simulate the dynamics using the SINDy model
        z_sim, dz_sim, ddz_sim = sindy_batch_simulate_order2(test_data['t'],
                                                            test_set_results['z'],
                                                            test_set_results['dz'],
                                                            test_set_results['u_latent'],
                                                            params['coefficient_mask'] * test_set_results['sindy_coefficients'],
                                                            params['library_type'],
                                                            params['poly_order'],
                                                            params['include_goniom'],
                                                            input_type='real',
                                                            batch_size=test_data['t'].shape[0])

        # Decode latent dynamics to original state space
        x_sim = decode_latent_representation(z_sim, autoencoder_network['decoder_weights'], autoencoder_network['decoder_biases'], params['activation'])
        dx_sim = decode_latent_representation(dz_sim, autoencoder_network['decoder_weights'], autoencoder_network['decoder_biases'], params['activation'])
        ddx_sim = decode_latent_representation(ddz_sim, autoencoder_network['decoder_weights'], autoencoder_network['decoder_biases'], params['activation'])

        # Evaluate decoded states
        x_sim_eval = sess.run(x_sim)
        dx_sim_eval = sess.run(dx_sim)
        ddx_sim_eval = sess.run(ddx_sim)

# Reset the TensorFlow graph to ensure it's ready for another run
tf.reset_default_graph()

# If the hybrid flag is set, load the second model and simulate the dynamics using the hybrid model
if hybrid:
    # Load model parameters for the hybrid model from a pickle file
    params_2 = pickle.load(open(data_path + save_name_2 + '_params.pkl', 'rb'))
    
    # Initialize the second autoencoder network with the loaded parameters
    autoencoder_network_2 = full_network(params_2)
    
    # Create a new placeholder for the learning rate in the second model
    learning_rate_2 = tf.placeholder(tf.float32, name='learning_rate_2')

    # Initialize a saver for the second autoencoder to manage TensorFlow variables
    saver_2 = tf.train.Saver()

    # Create an empty tuple to store TensorFlow operations for execution in the second model
    tensorflow_run_tuple_2 = ()
    # Populate the tuple with operations from the autoencoder network dictionary
    for key in autoencoder_network_2.keys():
        tensorflow_run_tuple_2 += (autoencoder_network_2[key],)

    # Identify indices where no feet of the robot are in contact with the ground
    flight_indices = np.where(np.sum(test_data['foot'], axis=1) == 0)[0]

    # Open a new TensorFlow session for the second model
    with tf.Session() as sess_2:
        # Initialize all global variables for the session
        sess_2.run(tf.global_variables_initializer())

        # Restore the trained model parameters from the specified save path
        saver_2.restore(sess_2, data_path + save_name_2)

        # Create a dictionary for feeding data to the TensorFlow session using parameters for the second model
        test_dictionary_2 = create_feed_dictionary(test_data, params_2)

        # Execute the TensorFlow operations defined in tensorflow_run_tuple_2
        tf_results_2 = sess_2.run(tensorflow_run_tuple_2, feed_dict=test_dictionary_2)

        # Collect the results from TensorFlow operations into a dictionary
        test_set_results_2 = {}
        for i, key in enumerate(autoencoder_network_2.keys()):
            test_set_results_2[key] = tf_results_2[i]

        # Simulate dynamics using the hybrid model based on test data and model parameters
        z_sim, dz_sim, ddz_sim = hybrid_sindy_batch_simulate_order2(test_data['t'],
                                                                    test_set_results['z'],
                                                                    test_set_results['dz'],
                                                                    test_set_results['u_latent'],
                                                                    flight_indices,
                                                                    params['coefficient_mask'] * test_set_results['sindy_coefficients'],
                                                                    params_2['coefficient_mask'] * test_set_results_2['sindy_coefficients'],
                                                                    params['library_type'],
                                                                    params['poly_order'],
                                                                    params['include_goniom'],
                                                                    input_type='real',
                                                                    batch_size=test_data['t'].shape[0]
                                                                    )
        
        # Decode latent states into observable states based on simulation results
        x_sim_list, dx_sim_list, ddx_sim_list = [], [], []
        for i in range(len(z_sim)):
            if i in flight_indices:
                x_sim_list.append(decode_latent_representation(z_sim[i].reshape(1, -1), test_set_results_2['decoder_weights'], test_set_results_2['decoder_biases'], params['activation']))
                dx_sim_list.append(decode_latent_representation(dz_sim[i].reshape(1, -1), test_set_results_2['decoder_weights'], test_set_results_2['decoder_biases'], params['activation']))
                ddx_sim_list.append(decode_latent_representation(ddz_sim[i].reshape(1, -1), test_set_results_2['decoder_weights'], test_set_results_2['decoder_biases'], params['activation']))
            else:
                x_sim_list.append(decode_latent_representation(z_sim[i].reshape(1, -1), test_set_results['decoder_weights'], test_set_results['decoder_biases'], params['activation']))
                dx_sim_list.append(decode_latent_representation(dz_sim[i].reshape(1, -1), test_set_results['decoder_weights'], test_set_results['decoder_biases'], params['activation']))
                ddx_sim_list.append(decode_latent_representation(ddz_sim[i].reshape(1, -1), test_set_results['decoder_weights'], test_set_results['decoder_biases'], params['activation']))

        # Convert lists to TensorFlow tensors for aggregation
        x_sim = tf.concat(x_sim_list, axis=0)
        dx_sim = tf.concat(dx_sim_list, axis=0)
        ddx_sim = tf.concat(ddx_sim_list, axis=0)

        # Evaluate the tensor operations to get numerical results
        x_sim_eval = sess_2.run(x_sim)
        dx_sim_eval = sess_2.run(dx_sim)
        ddx_sim_eval = sess_2.run(ddx_sim)

        # Simulate the same model for different time batches
        z_sim_b, dz_sim_b, ddz_sim_b = hybrid_sindy_batch_simulate_order2(test_data['t'],
                                                                          test_set_results['z'],
                                                                          test_set_results['dz'],
                                                                          test_set_results['u_latent'],
                                                                          flight_indices,
                                                                          params['coefficient_mask'] * test_set_results['sindy_coefficients'],
                                                                          params_2['coefficient_mask'] * test_set_results_2['sindy_coefficients'],
                                                                          params['library_type'],
                                                                          params['poly_order'],
                                                                          params['include_goniom'],
                                                                          input_type='real',
                                                                          batch_size=100
                                                                          )
        
        # Decode latent states into observable states based on simulation results
        x_sim_list_b, dx_sim_list_b, ddx_sim_list_b = [], [], []
        for i in range(len(z_sim_b)):
            if i in flight_indices:
                x_sim_list_b.append(decode_latent_representation(z_sim_b[i].reshape(1, -1), test_set_results_2['decoder_weights'], test_set_results_2['decoder_biases'], params['activation']))
                dx_sim_list_b.append(decode_latent_representation(dz_sim_b[i].reshape(1, -1), test_set_results_2['decoder_weights'], test_set_results_2['decoder_biases'], params['activation']))
                ddx_sim_list_b.append(decode_latent_representation(ddz_sim_b[i].reshape(1, -1), test_set_results_2['decoder_weights'], test_set_results_2['decoder_biases'], params['activation']))
            else:
                x_sim_list_b.append(decode_latent_representation(z_sim_b[i].reshape(1, -1), test_set_results['decoder_weights'], test_set_results['decoder_biases'], params['activation']))
                dx_sim_list_b.append(decode_latent_representation(dz_sim_b[i].reshape(1, -1), test_set_results['decoder_weights'], test_set_results['decoder_biases'], params['activation']))
                ddx_sim_list_b.append(decode_latent_representation(ddz_sim_b[i].reshape(1, -1), test_set_results['decoder_weights'], test_set_results['decoder_biases'], params['activation']))

        # Convert lists to TensorFlow tensors for aggregation
        x_sim_b = tf.concat(x_sim_list_b, axis=0)
        dx_sim_b = tf.concat(dx_sim_list_b, axis=0)
        ddx_sim_b = tf.concat(ddx_sim_list_b, axis=0)

        # Evaluate the tensor operations to get numerical results
        x_sim_eval_b = sess_2.run(x_sim_b)
        dx_sim_eval_b = sess_2.run(dx_sim_b)
        ddx_sim_eval_b = sess_2.run(ddx_sim_b)


### ASLIP SIMULATION ###
# Select only the 13th and 15th dimensions of the data
aslip_t = test_data['t']
aslip_pos = test_data['x'][:, [12, 14]] * 10
aslip_vel = test_data['dx'][:, [12, 14]] * 10
aslip_acc = test_data['ddx'][:, [12, 14]] * 10
aslip_u = test_data['u'][:, [12, 14]] * 10
# Simulate the aslip model
aslip_pos_sim, aslip_vel_sim, aslip_acc_sim = hybrid_aslip_batch_simulate(aslip_t, aslip_pos, aslip_vel, aslip_u, flight_indices, batch_size=test_data['t'].shape[0])

### RESULTS PLOTTING ###
# Give some metrics    
print('\nNumber of final active coefficients: ' + str(np.sum(params['coefficient_mask'])))
write_learnt_equations(params['coefficient_mask'] * test_set_results['sindy_coefficients'], params['latent_dim'], params['poly_order'], params['include_goniom'], params['input_type'])
print('\nFinal losses: ')
print('Decoder loss ratio: %f' % (np.array(results['loss_decoder']) / np.array(results['x_norm'])))
print('SINDy predict ddz loss ratio: %f' % (np.array(results['loss_sindy_ddz']) / np.array(results['sindy_predict_ddz_norm'])))
print('SINDy predict ddx loss ratio: %f' % (np.array(results['loss_sindy_ddx']) / np.array(results['sindy_predict_ddx_norm'])))

# Plot decoded data
# plot_joint_data(test_data['t'], test_data['x'][:, 0:12], test_set_results['x_decode'][:, 0:12])
# plot_body_data(test_data['t'], test_data['x'][:, 12:18], test_set_results['x_decode'][:, 12:18])

# Plot real joint data
# plot_joint_data(test_data['t'], test_data['x'][:, 0:12], x_sim_eval[:, 0:12])
# plot_joint_data(test_data['t'], test_data['dx'][:, 0:12], dx_sim_eval[:, 0:12])
# plot_joint_data(test_data['t'], test_data['ddx'][:, 0:12], ddx_sim_eval[:, 0:12])
plot_joint_data_3(test_data['t'], test_data['x'][:, 0:12], x_sim_eval[:, 0:12], save_filename=None)

# Plot real body data 
# plot_body_data(test_data['t'], test_data['x'][:, 12:18], x_sim_eval[:, 12:18])
# plot_body_data(test_data['t'], test_data['dx'][:, 12:18], dx_sim_eval[:, 12:18])
# plot_body_data(test_data['t'], test_data['ddx'][:, 12:18], ddx_sim_eval[:, 12:18])
plot_body_data_3(test_data['t'], test_data['x'][:, 12:18], None, x_sim_eval[:, 12:18], save_filename=None)

# Plot latent data
# plot_latent_data_2(test_data['t'], test_set_results['ddz'], ddz_sim, ddz_sim_b, save_filename=None)
# plot_latent_data_4(test_data['t'], test_set_results['ddz'], ddz_sim, ddz_sim_b, save_filename=None)

# Plot u
# plot_torque_data(test_data['t'], test_data['u'][:, 0:12])
# plot_forces_on_com(test_data['t'], test_data['u'][:, 12:18])

# Plot autoencoder weights heatmaps
encoder_weights = np.array(test_set_results['encoder_weights'])[0]
decoder_weights = np.array(test_set_results['decoder_weights'])[0]
encoder_decoder_weights = np.matmul(encoder_weights, decoder_weights)
# create_weights_heatmap(decoder_weights, 'decoder')
# create_weights_heatmap(encoder_weights, 'encoder')
# create_weights_heatmap(encoder_decoder_weights, 'encoder')

# Plot aslip data
# plot_aslip(aslip_t, aslip_pos, aslip_pos_sim)
# plot_aslip(aslip_t, aslip_acc, aslip_acc_sim)
# plot_aslip_3(aslip_t, aslip_pos, aslip_pos_sim, x_sim_eval[:, (12,14)], save_filename='aslip.png')

plt.show()