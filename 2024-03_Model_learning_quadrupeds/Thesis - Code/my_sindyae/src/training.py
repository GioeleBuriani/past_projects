# Import necessary libraries and modules
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow v2 behavior as the code is written for v1
import pickle

from autoencoder import full_network, define_loss
from sindy_utils import write_learnt_equations, write_learnt_aslip_equations

# Set a fixed seed for reproducibility of random processes
SEED = 0
tf.set_random_seed(SEED)

# Function to train the network, perform SINDy model identification, and handle model refinement
def train_network(training_data, val_data, params):
    # Create the autoencoder network from the parameters provided
    autoencoder_network = full_network(params)
    
    # Define the loss function with different components and the overall loss
    loss, losses, loss_refinement = define_loss(autoencoder_network, params)
    
    # Define a TensorFlow placeholder to dynamically adjust the learning rate
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # Conditionally set up the training operation based on whether the model is pretrained
    if not params['pretrained']:
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        train_op_refinement = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_refinement)
    else:
        # Allow training only specific parts of the model if it's pretrained
        trainable_variables = [var for var in tf.trainable_variables() if 'sindy' in var.name or 'decoder' in var.name]
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=trainable_variables)
        train_op_refinement = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_refinement, var_list=trainable_variables)
    
    # Initialize a TensorFlow Saver object to save the trained models
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    # Create feed dictionaries for training and validation data
    train_dict_print = create_feed_dictionary(training_data, params, idxs=None)
    validation_dict = create_feed_dictionary(val_data, params, idxs=None)

    # Calculate normalization factors for validation data used in loss calculations
    x_norm = np.mean(val_data['x']**2)
    sindy_predict_ddx_norm = np.mean(val_data['ddx']**2)

    # Initialize containers to hold training and validation losses and the terms in the SINDy model
    training_losses = []
    validation_losses = []
    sindy_model_terms = [np.sum(params['coefficient_mask'])]

    # Begin training process
    print('TRAINING')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # Initialize all global variables
        
        # Main training loop
        for i in range(params['max_epochs']):
            for j in range(params['epoch_size'] // params['batch_size']):
                batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                
                sess.run(train_op, feed_dict=train_dict)  # Execute a single training step

            # Periodically output training progress and compute validation losses
            if params['print_progress'] and (i % params['print_frequency'] == 0):
                training_loss, validation_loss = print_progress(sess, i, loss, losses, train_dict_print, validation_dict, x_norm, sindy_predict_ddx_norm)
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)

            # Apply sequential thresholding to prune small SINDy coefficients
            if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
                params['coefficient_mask'] = np.abs(sess.run(autoencoder_network['sindy_coefficients'])) > params['coefficient_threshold']
                train_dict['coefficient_mask:0'] = params['coefficient_mask']
                validation_dict['coefficient_mask:0'] = params['coefficient_mask']
                print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
                sindy_model_terms.append(np.sum(params['coefficient_mask']))

        # Refinement phase to fine-tune the model
        print('REFINEMENT')
        for i_refinement in range(params['refinement_epochs']):
            for j in range(params['epoch_size'] // params['batch_size']):
                batch_idxs = np.arange(j * params['batch_size'], (j + 1) * params['batch_size'])
                train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                sess.run(train_op_refinement, feed_dict=train_dict)
            
            if params['print_progress'] and (i_refinement % params['print_frequency'] == 0):
                training_loss, validation_loss = print_progress(sess, i_refinement, loss_refinement, losses, train_dict_print, validation_dict, x_norm, sindy_predict_ddx_norm)
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)

        # Save the trained model and dump parameters to a pickle file
        saver.save(sess, params['data_path'] + params['save_name'])
        pickle.dump(params, open(params['data_path'] + params['save_name'] + '_params.pkl', 'wb'))

        # Gather and report final model evaluation metrics
        final_losses = sess.run((losses['decoder'], losses['sindy_z'], losses['sindy_ddz'], losses['sindy_ddx'], losses['sindy_regularization']), feed_dict=validation_dict)
        sindy_predict_ddz_norm = np.mean(sess.run(autoencoder_network['ddz'], feed_dict=validation_dict)**2)
        sindy_coefficients = sess.run(autoencoder_network['sindy_coefficients'], feed_dict={})
        
        # Print learned SINDy models depending on the library type
        if params['library_type'] == 'general':
            write_learnt_equations(params['coefficient_mask'] * sindy_coefficients, latent_dim=params['latent_dim'], poly_order=params['poly_order'], include_goniom=params['include_goniom'], input_type=params['input_type'])
        elif params['library_type'] == 'aslip':
            write_learnt_aslip_equations(params['coefficient_mask'] * sindy_coefficients)

        # Organize and return results
        results_dict = {
            'num_epochs': i,
            'x_norm': x_norm,
            'sindy_predict_ddx_norm': sindy_predict_ddx_norm,
            'sindy_predict_ddz_norm': sindy_predict_ddz_norm,
            'sindy_coefficients': sindy_coefficients,
            'loss_decoder': final_losses[0],
            'loss_sindy_z': final_losses[1],
            'loss_sindy_ddz': final_losses[2],
            'loss_sindy_ddx': final_losses[3],
            'loss_sindy_regularization': final_losses[4],
            'training_losses': np.array(training_losses),
            'validation_losses': np.array(validation_losses),
            'sindy_model_terms': np.array(sindy_model_terms),
            'encoder_weights': sess.run(autoencoder_network['encoder_weights']),
            'encoder_biases': sess.run(autoencoder_network['encoder_biases']),
            'decoder_weights': sess.run(autoencoder_network['decoder_weights']),
            'decoder_biases': sess.run(autoencoder_network['decoder_biases'])
        }
        return results_dict

# Function to create a TensorFlow feed dictionary from data and parameters, optionally for specific indices
def create_feed_dictionary(data, params, idxs=None):
    if idxs is None:
        idxs = np.arange(data['x'].shape[0])  # Default to all indices if none are provided

    feed_dict = {}
    feed_dict['x:0'] = data['x'][idxs]  # Input state variables
    feed_dict['dx:0'] = data['dx'][idxs]  # First derivatives of state variables
    
    # Include second derivatives in the feed dictionary if the model order is 2
    feed_dict['ddx:0'] = data['ddx'][idxs]

    feed_dict['u:0'] = data['u'][idxs]  # Control inputs
    
    # Include the coefficient mask for use in loss functions and training, if using sequential thresholding
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
        
    feed_dict['learning_rate:0'] = params['learning_rate']  # Current learning rate

    return feed_dict  # Return the fully formed dictionary

# Function to print training and validation loss progress during network training
def print_progress(sess, i, loss, losses, train_dict, validation_dict, x_norm, sindy_predict_norm):
    # Compute both training and validation loss values from the TensorFlow session
    training_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=train_dict)
    validation_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=validation_dict)
    
    # Output current epoch and associated loss metrics
    print("Epoch %d" % i)
    print("   training loss {0}, {1}".format(training_loss_vals[0], training_loss_vals[1:]))
    print("   validation loss {0}, {1}".format(validation_loss_vals[0], validation_loss_vals[1:]))

    # Compute and print loss ratios to assess model performance relative to normalization factors
    decoder_losses = sess.run((losses['decoder'], losses['sindy_ddx']), feed_dict=validation_dict)
    loss_ratios = (decoder_losses[0] / x_norm, decoder_losses[1] / sindy_predict_norm)
    print("decoder loss ratio: %f, decoder SINDy loss ratio: %f" % loss_ratios)
    
    return training_loss_vals, validation_loss_vals  # Return computed loss values for further analysis
