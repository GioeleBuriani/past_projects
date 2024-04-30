# Import TensorFlow v1 compatibility mode and numpy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import pickle

# Set a random seed for reproducibility
SEED = 0
tf.set_random_seed(SEED)

# Function to build the full neural network model with specified parameters
def full_network(params):   

    # Extract network parameters from the 'params' dictionary
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']

    input_type = params['input_type']
    poly_order = params['poly_order']
    if 'include_goniom' in params.keys():
        include_goniom = params['include_goniom']
    else:
        include_goniom = False
    library_dim = params['library_dim']

    network = {}

    # Initialize placeholders for input data and derivatives
    x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
    dx = tf.placeholder(tf.float32, shape=[None, input_dim], name='dx')
    ddx = tf.placeholder(tf.float32, shape=[None, input_dim], name='ddx')
    u = tf.placeholder(tf.float32, shape=[None, input_dim], name='u')

    # Define the encoder and decoder based on the activation type
    if activation == 'linear':
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(x, input_dim, latent_dim, params['pretrained'])
    else:
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(x, input_dim, latent_dim, params['widths'], params['pretrained'], activation=activation)

    # Compute the Jacobian of the encoder output
    encoder_jacobian = compute_jacobian(z, x)
    transposed_encoder_jacobian = tf.transpose(encoder_jacobian, perm=[0, 2, 1])
    transposed_inverted_encoder_jacobian = tf.linalg.pinv(transposed_encoder_jacobian)
    u_expanded = tf.expand_dims(u, axis=2)
    result_matrix = tf.matmul(transposed_inverted_encoder_jacobian, u_expanded)
    u_latent = tf.squeeze(result_matrix, axis=2)

    # Calculate derivatives using the autoencoder encoder
    dz, ddz = z_derivative_order2(x, dx, ddx, encoder_weights, encoder_biases, activation=activation)
    
    # Construct the SINDy library based on the model configuration
    if params['library_type'] == 'general':
        Theta = sindy_library_tf_order2(z, dz, u_latent, latent_dim, poly_order, include_goniom, input_type)
    elif params['library_type'] == 'aslip':
        Theta = sindy_library_aslip_tf(z, dz, u_latent)

    # Initialize SINDy coefficients based on specified initialization method
    if params['coefficient_initialization'] == 'xavier':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim, latent_dim], initializer=tf.keras.initializers.glorot_normal())
    elif params['coefficient_initialization'] == 'specified':
        sindy_coefficients = tf.get_variable('sindy_coefficients', initializer=params['init_coefficients'])
    elif params['coefficient_initialization'] == 'constant':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim, latent_dim], initializer=tf.constant_initializer(1.0))
    elif params['coefficient_initialization'] == 'normal':
        sindy_coefficients = tf.get_variable('sindy_coefficients', shape=[library_dim, latent_dim], initializer=tf.initializers.random_normal())
    elif params['coefficient_initialization'] == 'pretrained':
        data_path = os.getcwd() + '/my_sindyae/quadrupeds/Results/'
        save_name = ''
        results = pickle.load(open(data_path + save_name + '_experiment_results.pkl', 'rb'))
        sindy_coefficients = tf.get_variable('sindy_coefficients', initializer=np.array(results['sindy_coefficients'])[0])

    # Compute SINDy predictions based on the presence of sequential thresholding
    if params['sequential_thresholding']:
        coefficient_mask = tf.placeholder(tf.float32, shape=[library_dim, latent_dim], name='coefficient_mask')
        sindy_predict = tf.matmul(Theta, coefficient_mask*sindy_coefficients)
        network['coefficient_mask'] = coefficient_mask
    else:
        sindy_predict = tf.matmul(Theta, sindy_coefficients)

    # Check if backpropagation through time is enabled in the parameters
    # NOTE: BPTT hasn't been tested with the newest version of the code therefore if activated might create issues
    if params['BPTT']:
        # Retrieve necessary parameters for BPTT
        prediction_window_size = params['prediction_window']
        latent_dim = params['latent_dim']
        integration_method = params['integration_method']
        dt = params['integration_tstep']

        # Initialize future_predictions to store each state's prediction for the window size
        future_predictions = tf.tile(tf.expand_dims(z, 1), [1, prediction_window_size, 1])

        # Set initial states for position and velocity
        current_position = z
        current_velocity = dz
        sindy_predict_next = sindy_predict

        # Iterate over each time step within the prediction window
        for i in range(1, prediction_window_size):
            if integration_method == 'Euler':
                # Euler method for numerical integration to update position and velocity
                current_position += dt * current_velocity
                current_velocity += dt * sindy_predict_next
                
                # Calculate SINDy library for the current state to get next state predictions
                Theta_next = sindy_library_tf_order2(current_position, current_velocity, u_latent, latent_dim, poly_order, include_goniom, input_type)
                sindy_predict_next = tf.matmul(Theta_next, sindy_coefficients)

            elif integration_method == 'RK2':

                # WARNING: For this method dt has to be modified to .01 both for the integration and the data generation to be consistent

                # Note regarding timestep adjustment for consistency in RK2 integration
                k1 = dt * sindy_predict_next

                # Midpoint estimation for RK2
                Theta_mid = sindy_library_tf_order2(current_position + 0.5 * dt * current_velocity,
                                                    current_velocity + 0.5 * k1, u_latent, latent_dim, poly_order, include_goniom, input_type)
                k2 = dt * tf.matmul(Theta_mid, sindy_coefficients)

                # Update position and velocity using RK2 step
                current_position += dt * current_velocity
                current_velocity += k2

                # Compute SINDy library for next state prediction
                Theta_next = sindy_library_tf_order2(current_position, current_velocity, u_latent, latent_dim, poly_order, include_goniom, input_type)
                sindy_predict_next = tf.matmul(Theta_next, sindy_coefficients)

            # Calculate the maximum number of rows that can be updated at this time step
            number_of_rows_to_store = tf.shape(future_predictions)[0] - i

            # Extract predictions for positions to be updated
            updates = current_position[:number_of_rows_to_store, :]
            flattened_updates = tf.reshape(updates, [-1])

            # Indices for tensor updates
            start_row_index = i
            batch_indices = tf.range(start_row_index, start_row_index + number_of_rows_to_store)
            time_indices = tf.ones_like(batch_indices) * i
            batch_indices = tf.repeat(batch_indices, latent_dim)
            time_indices = tf.repeat(time_indices, latent_dim)
            latent_dim_indices = tf.tile(tf.range(latent_dim), [number_of_rows_to_store])

            indices_to_update = tf.stack([batch_indices, time_indices, latent_dim_indices], axis=-1)
            indices_to_update = tf.reshape(indices_to_update, [-1, 3])

            # Update future predictions with current predictions at calculated indices
            future_predictions = tf.tensor_scatter_nd_update(future_predictions, indices_to_update, flattened_updates)

        # Store the calculated future predictions in the network for further usage
        network['future_predictions'] = future_predictions

    # Calculate the reconstructed derivatives using the decoded predictions
    dx_predict, ddx_predict = z_derivative_order2(z, dz, sindy_predict, decoder_weights, decoder_biases, activation=activation)

    # Store all relevant TensorFlow objects in the 'network' dictionary
    network['x'] = x
    network['dx'] = dx
    network['ddx'] = ddx
    network['u'] = u

    network['x_decode'] = x_decode
    
    network['z'] = z
    network['dz'] = dz
    network['ddz'] = ddz
    network['u_latent'] = u_latent

    network['ddz_predict'] = sindy_predict

    network['dx_predict'] = dx_predict
    network['ddx_predict'] = ddx_predict

    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases

    network['Theta'] = Theta
    network['sindy_coefficients'] = sindy_coefficients

    return network

# Function to build a simple linear autoencoder
def linear_autoencoder(x, input_dim, latent_dim, pretrained):
    # Build the encoder portion of the autoencoder to compress input 'x' to latent space 'z'
    z, encoder_weights, encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder', pretrained)
    
    # Build the decoder portion of the autoencoder to reconstruct the input from latent space 'z' back to 'x_decode'
    x_decode, decoder_weights, decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder', pretrained)

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases

# Function to build a more complex autoencoder using nonlinear activation functions
def nonlinear_autoencoder(x, input_dim, latent_dim, widths, pretrained, activation='elu'):
    # Map the activation string to the corresponding TensorFlow activation function
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('Invalid activation function specified.')

    # Build the encoder using specified widths and activation function to compress input 'x' to latent space 'z'
    z, encoder_weights, encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder', pretrained)
    
    # Build the decoder using reversed widths and the same activation function to reconstruct the input from 'z' back to 'x_decode'
    x_decode, decoder_weights, decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder', pretrained)

    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases

# Function to construct individual network layers
def build_network_layers(input, input_dim, output_dim, widths, activation, name, pretrained):
    # Initialize lists to store the weights and biases of each layer
    weights = []
    biases = []

    # Set the initial layer width to the input dimension
    last_width = input_dim

    # Load pretrained models if applicable
    if pretrained:
        data_path = os.getcwd() + '/my_sindyae/go1/Results/'
        save_name = ''
        results = pickle.load(open(data_path + save_name + '_experiment_results.pkl', 'rb'))

    # Construct each layer in the network
    for i, n_units in enumerate(widths):
        if not pretrained:
            # Initialize weights with Xavier/Glorot uniform initializer and biases to zeros if not using pretrained weights
            W = tf.get_variable(name+'_W'+str(i), shape=[last_width, n_units], initializer=tf.keras.initializers.glorot_normal())
            b = tf.get_variable(name+'_b'+str(i), shape=[n_units], initializer=tf.constant_initializer(0.0))
        else:
            # Use weights and biases from pretrained model if specified
            if name == 'encoder' or name == 'decoder':
                W = tf.get_variable(name+'_W'+str(i), initializer=np.array(results[name + '_weights'])[i])
                b = tf.get_variable(name+'_b'+str(i), initializer=np.array(results[name + '_biases'])[i])

        # Apply the weights and biases to the input, and use the activation function if one is provided
        input = tf.matmul(input, W) + b
        if activation is not None:
            input = activation(input)

        # Update the dimension tracker and store the current layer's weights and biases
        last_width = n_units
        weights.append(W)
        biases.append(b)

    # Create the final output layer with appropriate weights and biases, also checking for pretrained settings
    if not pretrained:
        W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width, output_dim], initializer=tf.keras.initializers.glorot_normal())
        b = tf.get_variable(name+'_b'+str(len(widths)), shape=[output_dim], initializer=tf.constant_initializer(0.0))
    else:
        W = tf.get_variable(name+'_W'+str(len(widths)), initializer=np.array(results[name + '_weights'])[-1])
        b = tf.get_variable(name+'_b'+str(len(widths)), initializer=np.array(results[name + '_biases'])[-1])

    # Process the final layer output
    input = tf.matmul(input, W) + b
    weights.append(W)
    biases.append(b)

    # Return the output, weights, and biases for chaining and retrieval
    return input, weights, biases

# Function to compute the Jacobian matrix of a network output with respect to its input
def compute_jacobian(network_output, network_input):
    output_dim = int(network_output.shape[1])  # Determine the number of output dimensions
    
    # Compute the gradient of each output dimension with respect to the input
    jacobian_rows = [tf.gradients(network_output[:, i], network_input)[0] for i in range(output_dim)]
    # Stack the computed gradients to form the Jacobian matrix
    jacobian = tf.stack(jacobian_rows, axis=1)  # Shape: [n_timesteps, output_dim, input_dim]
    
    return jacobian

# Function to compute the first and second derivatives of the network's output
def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    dz = dx  # Initialize the first derivative of the latent state
    ddz = ddx  # Initialize the second derivative of the latent state

    # Apply the appropriate derivative computation based on the activation function
    if activation == 'elu':
        for i in range(len(weights)-1):
            # Propagate the input through the network layer
            input = tf.matmul(input, weights[i]) + biases[i]
            # Compute first and second derivatives using ELU activation properties
            dz_prev = tf.matmul(dz, weights[i])
            elu_derivative = tf.minimum(tf.exp(input), 1.0)  # Derivative of ELU for the first order
            elu_derivative2 = tf.multiply(tf.exp(input), tf.to_float(input < 0))  # Second order derivative
            dz = tf.multiply(elu_derivative, dz_prev)
            ddz = tf.multiply(elu_derivative2, tf.square(dz_prev)) + tf.multiply(elu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.elu(input)

    elif activation == 'relu':
        for i in range(len(weights)-1):
            # Propagate the input and update derivatives for ReLU activation
            input = tf.matmul(input, weights[i]) + biases[i]
            relu_derivative = tf.to_float(input > 0)  # Derivative of ReLU
            dz = tf.multiply(relu_derivative, tf.matmul(dz, weights[i]))
            ddz = tf.multiply(relu_derivative, tf.matmul(ddz, weights[i]))
            input = tf.nn.relu(input)

    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            # Propagate the input and update derivatives for sigmoid activation
            input = tf.matmul(input, weights[i]) + biases[i]
            input = tf.sigmoid(input)
            dz_prev = tf.matmul(dz, weights[i])
            sigmoid_derivative = tf.multiply(input, 1 - input)  # Derivative of sigmoid
            sigmoid_derivative2 = tf.multiply(sigmoid_derivative, 1 - 2 * input)  # Second order derivative
            dz = tf.multiply(sigmoid_derivative, dz_prev)
            ddz = tf.multiply(sigmoid_derivative2, tf.square(dz_prev)) + tf.multiply(sigmoid_derivative, tf.matmul(ddz, weights[i]))

    else:
        # Default derivative computation when no activation is specified
        for i in range(len(weights)-1):
            dz = tf.matmul(dz, weights[i])
            ddz = tf.matmul(ddz, weights[i])

    # Finalize the computation of derivatives at the output layer
    dz = tf.matmul(dz, weights[-1])
    ddz = tf.matmul(ddz, weights[-1])

    return dz, ddz

# Function to build a SINDy library with polynomial and optional trigonometric functions up to a specified order
def sindy_library_tf_order2(z, dz, u_latent, latent_dim, poly_order, include_goniom=False, input_type=None):
    # Start the library with a column of ones for bias
    library = [tf.ones(tf.shape(z)[0], dtype=tf.float32)]

    # Combine state variables z and their derivatives dz for library construction
    z_combined = tf.concat([z, dz], 1)

    # Add first order terms: state variables and their derivatives
    for i in range(2 * latent_dim):
        library.append(z_combined[:, i])

    # Generate polynomial terms up to the specified order
    if poly_order > 1:
        # Second-order terms
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                library.append(tf.multiply(z_combined[:, i], z_combined[:, j]))

    if poly_order > 2:
        # Third-order terms
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k])

    if poly_order > 3:
        # Fourth-order terms
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    for p in range(k, 2 * latent_dim):
                        library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k] * z_combined[:, p])

    if poly_order > 4:
        # Fifth-order terms
        for i in range(2 * latent_dim):
            for j in range(i, 2 * latent_dim):
                for k in range(j, 2 * latent_dim):
                    for p in range(k, 2 * latent_dim):
                        for q in range(p, 2 * latent_dim):
                            library.append(z_combined[:, i] * z_combined[:, j] * z_combined[:, k] * z_combined[:, p] * z_combined[:, q])

    # Include sine and cosine functions if specified
    if include_goniom:
        for i in range(2 * latent_dim):
            library.append(tf.sin(z_combined[:, i]))
            library.append(tf.cos(z_combined[:, i]))

    # Include control inputs in the library if applicable
    if input_type is not None:
        for i in range(latent_dim):
            library.append(u_latent[:, i])

    # Stack the library terms along columns to create the library matrix
    return tf.stack(library, axis=1)

# Function to construct a SINDy library specifically for a model of a spring-loaded inverted pendulum (ASLIP)
def sindy_library_aslip_tf(z, dz, u_latent):
    # Extract specific state variables and control inputs
    x = z[:, 0]  # Position along x-axis
    z_height = z[:, 1]  # Position along z-axis
    u_x = u_latent[:, 0]  # Control input along x-axis
    u_z = u_latent[:, 1]  # Control input along z-axis

    # Initialize the library with a constant column for bias
    library = [tf.ones(tf.shape(z_height)[0], dtype=tf.float32)]

    # Add state variables and control inputs to the library
    library.append(x)
    library.append(z_height)
    library.append(u_x)
    library.append(u_z)

    # Stack the library terms along columns to create the library matrix
    return tf.stack(library, axis=1)

# Function to define various loss components for training the network
def define_loss(network, params):
    # Extract relevant tensors from the network dictionary
    x = network['x']  # Original input data
    x_decode = network['x_decode']  # Autoencoder's reconstructed input

    ddz = network['ddz']  # Actual second-order derivative of the latent representation
    ddz_predict = network['ddz_predict']  # Predicted second-order derivative from SINDy
    ddx = network['ddx']  # Actual second derivative of the input data
    ddx_predict = network['ddx_predict']  # Reconstructed second derivative from the model

    # Apply coefficient mask to SINDy coefficients for regularization
    sindy_coefficients = params['coefficient_mask'] * network['sindy_coefficients']

    # Dictionary to hold individual components of loss
    losses = {}
    
    # Mean squared error loss between actual input and reconstructed input
    losses['decoder'] = tf.reduce_mean((x - x_decode)**2)

    # Loss computation for backpropagation through time enabled scenario
    if params['BPTT']:
        # Calculate absolute errors for future state predictions
        errors = tf.abs(network['future_predictions'] - tf.expand_dims(network['z'], 1))

        # Exponential decay factor for future predictions
        decay_factor = params['weight_decay_factor']
        prediction_window_dim = tf.shape(network['future_predictions'])[1]
        weights = tf.pow(decay_factor, tf.range(prediction_window_dim, dtype=tf.float32))

        # Weight errors by the exponential decay factor
        weighted_errors = errors * tf.reshape(weights, [1, -1, 1])
        sum_weighted_errors = tf.reduce_sum(weighted_errors, axis=-1)

        # Mean loss over all predicted future states
        losses['sindy_z'] = tf.reduce_mean(sum_weighted_errors)

        # Mean squared error for predicted second-order derivative in the latent space
        losses['sindy_ddz'] = tf.reduce_mean((ddz - ddz_predict)**2)

    else:
        # Zero loss when BPTT is not enabled
        losses['sindy_z'] = tf.reduce_mean(tf.zeros_like(network['z']))

        # Loss for second-order derivative prediction accuracy
        losses['sindy_ddz'] = tf.reduce_mean((ddz - ddz_predict)**2)

    # Loss for second-order derivative of input prediction accuracy
    losses['sindy_ddx'] = tf.reduce_mean((ddx - ddx_predict)**2)

    # L1 regularization loss for SINDy coefficients
    losses['sindy_regularization'] = tf.reduce_mean(tf.abs(sindy_coefficients))
    
    # Weighted sum of all individual losses
    loss = params['loss_weight_decoder'] * losses['decoder'] \
         + params['loss_weight_sindy_z'] * losses['sindy_z'] \
         + params['loss_weight_sindy_ddz'] * losses['sindy_ddz'] \
         + params['loss_weight_sindy_ddx'] * losses['sindy_ddx'] \
         + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    # Loss function for model refinement, excluding SINDy coefficient regularization
    loss_refinement = params['loss_weight_decoder'] * losses['decoder'] \
                    + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                    + params['loss_weight_sindy_ddz'] * losses['sindy_ddz'] \
                    + params['loss_weight_sindy_ddx'] * losses['sindy_ddx']

    return loss, losses, loss_refinement

# Function to reconstruct data from latent representation using the decoder part of the network
def decode_latent_representation(z_sim, decoder_weights, decoder_biases, activation):
    # Map activation function name to TensorFlow function
    if activation == 'linear':
        activation_function = None
    elif activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    else:
        raise ValueError('Invalid activation function')

    # Prepare input tensor from the normalized latent representation
    input = np.float32(z_sim)
    
    # Iterate over decoder layers to apply weights and biases
    for i in range(len(decoder_weights)):
        W = decoder_weights[i]
        b = decoder_biases[i]
        input = tf.matmul(input, W) + b
        if i < len(decoder_weights) - 1 and activation_function:  # Apply activation function except on the output layer
            input = activation_function(input)

    # Final output after passing through all layers
    x_reconstructed = input

    return x_reconstructed