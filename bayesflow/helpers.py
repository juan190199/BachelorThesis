def apply_gradients(optimizer, gradients, variables, global_step=None):
    """
    Performs one step of the backprop algorithm by updating each tensor in the 'variables' list.
    Note, that the operation is performed in-place.

    :param optimizer: tf.train.Optimizer
        Optimizer instance supporting an apply_gradeints() method

    :param gradients: list of tf.Tensor
        List of gradients for all neural network parameter

    :param variables: list of tf.Tensor
        List of all neural network parameters

    :param global_step: tf.Variable
        Integer valued tf.Variable indicating the current iteration step

    :return:
    """

    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
