import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

np.set_printoptions(suppress=True)


class Permutation(tf.keras.Model):
    """
    Implements a permutation layer to permute the input dimensions of the cINN block.
    """

    def __init__(self, input_dim):
        """
        Creates a permutation layer for a conditional invertible block.

        :param input_dim: int
            Dimensionality of the input to the c inv block.
        """
        super(Permutation, self).__init__()

        permutation_vec = np.random.permutation(input_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(initial_value=permutation_vec,
                                       trainable=False,
                                       dtype=tf.int32,
                                       name='permutation')
        self.inv_permutation = tf.Variable(initial_value=inv_permutation_vec,
                                           trainable=False,
                                           dtype=tf.int32,
                                           name='inv_permutation')

    def call(self, x, inverse=False):
        """
        Permutes the batch of an input.
        """

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(x), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(x), self.inv_permutation))


class CouplingNet(tf.keras.Model):
    """
    Implements a conditional version of a sequential network.
    """

    def __init__(self, meta, n_out):
        """
        Creates a conditional coupling net (FC neural network).

        :param meta: list
            List of dictionaries, wherein each dictionary holds parameter-value pairs for a single tf.keras.Dense layer.

        :param n_out: int
            Number of outputs of the coupling net
        """
        super(CouplingNet, self).__init__()

        self.dense = tf.keras.Sequential(
            # Hidden layer structure
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']] +
            # Output layer
            [tf.keras.layers.Dense(n_out,
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))]
        )

    def call(self, theta, x):
        """
        Concatenates x and y and performs a forward pass through the coupling net.

        :param theta: tf.Tensor of shape (batch_size, inp_dim)
            Parameters x ~ p(x|y) of interest

        :param x: tf.Tensor of shape (batch_size, summary_dim)
            Summarized conditional data of interest y = sum(y)

        :return:
        """
        inp = tf.concat((theta, x), axis=-1)
        out = self.dense(inp)
        return out


class HeteroskedasticModel(tf.keras.Model):
    """

    """

    def __init__(self, meta, summary_net=None):
        """
        Initializes custom model

        :param meta: dictionary
            Dictionary indicating structure of inference net

        :param summary_net:
        """
        super(HeteroskedasticModel, self).__init__()

        self.summary_net = summary_net
        self.inference_net = tf.keras.Sequential(
            [tf.keras.layers.Dense(
                units,
                activation=meta['activation'],
                kernel_initializer=meta['initializer'],
                kernel_regularizer=l2(meta['w_decay']))
                for units in meta['n_units']] +

            [tf.keras.layers.Dense(10,
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))]
        )

    def call(self, x):
        if self.summary_net is not None:
            y_summary = self.summary_net(x)
            y_pred = self.inference_net(y_summary)
            return y_pred
        else:
            y_pred = self.inference_net(x)
            return y_pred


class SequenceNet(tf.keras.Model):

    def __init__(self):
        """
        Creates a custom summary network, a combination of 1D conv and LSTM.
        """
        super(SequenceNet, self).__init__()

        self.conv_part = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, 3, activation='elu'),
            tf.keras.layers.Conv1D(64, 3, 3, activation='elu'),
            tf.keras.layers.Conv1D(64, 3, 3, activation='elu'),
            tf.keras.layers.GlobalAveragePooling1D()
        ])

        self.lstm_part = Sequential(
            [LSTM(32, return_sequences=True),
             LSTM(64)
             ])

    def call(self, x):
        """
        Performs a forward pass.
        """
        conv_out = self.conv_part(x)
        lstm_out = self.lstm_part(x)
        out = tf.concat((conv_out, lstm_out), axis=-1)
        return out
