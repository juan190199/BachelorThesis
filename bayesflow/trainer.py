import tensorflow as tfimport numpy as npfrom seird.model import data_generator# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch# https://www.tensorflow.org/tutorials/customization/custom_training_walkthroughdef train_step(model, optimizer, loss_fn, iterations, batch_size, p_bar=None, global_step=None, version='v3', S=False, E=False, learning_noise=False):    """    Performs a number of training iterations with a given tensorflow model and optimizer    :param model: tf.keras.Model -- a neural network model implementing a __call__() method    :param optimizer: tf.train.Optimizer -- the optimizer used for backprop    :param data_gen: callable -- a function providing batches of data    :param loss_fun: callable -- a function computing the loss given model outputs    :param iterations: int -- the number of training loops to perform    :param batch_size: int -- the batch_size used for training    :param p_bar: ProgressBar or None -- an instance for tracking the training progress    :param clip_value: float -- the value used for clipping the gradients    :param clip_method: str -- the method used for clipping (default 'global_norm')    :param global_step: tf.Variavle -- a scalar tensor tracking the number of steps and used for learning rate decay    :return: losses -- a dictionary with regularization and loss evaluations at each training iteration    """    # Prepare a dictionary for storing losses    losses = []    # Run training loop    for iteration in range(1, iterations + 1):        # Generate inputs for the network        try:            batch = data_generator(n_samples=batch_size, version=version, S=S, E=E)            if version == 'v5':                X, noisy_X, params = batch['X'], batch['noisy_X'], batch['params']            elif version == 'v6':                X, dropped_X, params = batch['X'], batch['dropped_X'], batch['params']            else:                X, params = batch['X'], batch['params']        except RuntimeError:            print("Runtime warning, skipping batch...")            p_bar.update(1)            continue        except FloatingPointError:            print("Floating point error, skipping batch...")            p_bar.update(1)            continue        with tf.GradientTape() as tape:            # Run the forward pass of the layer.            # The operations that the layer applies            # to its inputs are going to be recorded            # on the GradientTape.            # Forward pass            if version == 'v5':                if learning_noise:                    y_pred = model(params, noisy_X)                else:                    y_pred = model(params[:, :-1], noisy_X)            elif version == 'v6':                y_pred = model(params, dropped_X)            else:                y_pred = model(params, X)            # Loss computation            loss = loss_fn(y_pred['z'], y_pred['log_det_J'])        # One step backpropagation        grads = tape.gradient(loss, model.trainable_weights)        # Performs one step of the backpropagation algorithm by updating each tensor in the 'variables' list.        optimizer.apply_gradients(zip(grads, model.trainable_weights))        # Store losses        losses.append(loss)        # Update progress bar        if p_bar is not None:            p_bar.set_postfix_str("Iteration: {0}, Loss: {1}".format(iteration, loss))            p_bar.update(1)    return losses