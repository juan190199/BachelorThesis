import numpy as np
import tensorflow as tf

from bayesflow.networks import CouplingNet, Permutation


class ConditionalInvertibleBlock(tf.keras.Model):
    """
    Implements a conditional version of the INN block.
    """

    def __init__(self, meta, theta_dim, alpha=1.9, permute=False):
        """
        Creates a conditional invertible block.
        :param meta: list -- a list of dictionaries, wherein each dictionary holds parameter -
        value pairs for a single tf.keras.Dense layer. All coupling nets are assumed to be equal.
        :param theta_dim: int  -- the number of outputs of the invertible block
        (eq. the dimensionality of the latent space)
        :param alpha:
        :param permute:
        """
        super(ConditionalInvertibleBlock, self).__init__()

        self.alpha = alpha
        self.n_out1 = theta_dim // 2
        self.n_out2 = theta_dim // 2 if theta_dim % 2 == 0 else theta_dim // 2 + 1
        if permute:
            self.permutation = Permutation(theta_dim)
        else:
            self.permutation = None
        self.s1 = CouplingNet(meta, self.n_out1)
        self.t1 = CouplingNet(meta, self.n_out1)
        self.s2 = CouplingNet(meta, self.n_out2)
        self.t2 = CouplingNet(meta, self.n_out2)

    def call(self, theta, x, inverse=False, log_det_J=True):
        """
        Implements both directions of a conditional invertible block.
        :param theta: tf.Tensor of shape (batch_size, theta_dim) -- the parameters theta ~ p(theta|y) of interest
        :param x: tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest x = sum(x)
        :param inverse: bool -- flag indicating whether to tun the block forward or backwards
        :param log_det_J: bool -- flag indicating whether to return the log determinant of the Jacobian matrix
        :return:
        (v, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) -- the
        transformed input, if inverse = False, and the corresponding Jacobian of the transformation if inverse = False
        u               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """
        # --- Forward pass --- #
        if not inverse:

            if self.permutation is not None:
                theta = self.permutation(theta)

            u1, u2 = tf.split(theta, [self.n_out1, self.n_out2], axis=-1)

            # Pre-compute network outputs for v1
            s1 = self.s1(u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            t1 = self.t1(u2, x)
            v1 = u1 * tf.exp(s1) + t1

            # Pre-compute network outputs for v2
            s2 = self.s2(v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            t2 = self.t2(v1, x)
            v2 = u2 * tf.exp(s2) + t2
            v = tf.concat((v1, v2), axis=-1)

            if log_det_J:
                # log|J| = log(prod(diag(J))) -> according to inv architecture
                return v, tf.reduce_sum(s1, axis=-1) + tf.reduce_sum(s2, axis=-1)
            return v

        # --- Inverse pass --- #
        else:

            v1, v2 = tf.split(theta, [self.n_out1, self.n_out2], axis=-1)

            # Pre-Compute s2
            s2 = self.s2(v1, x)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            u2 = (v2 - self.t2(v1, x)) * tf.exp(-s2)

            # Pre-Compute s1
            s1 = self.s1(u2, x)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            u1 = (v1 - self.t1(u2, x)) * tf.exp(-s1)
            u = tf.concat((u1, u2), axis=-1)

            if self.permutation is not None:
                u = self.permutation(u, inverse=True)
            return u


class BayesFlow(tf.keras.Model):
    """
    Implements a chain of conditional invertible blocks for Bayesian parameter inference.
    """

    def __init__(self, meta, n_blocks, theta_dim, alpha=1.9, summary_net=None, permute=False):
        """

        :param meta: list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
        keras.Dense layer
        :param n_blocks: int  -- the number of invertible blocks
        :param theta_dim: int  -- the dimensionality of the parameter space to be learned
        :param alpha:
        :param summary_net: tf.keras.Model or None -- an optinal summary network for learning the sumstats of x
        :param permute: bool -- whether to permute the inputs to the cINN
        """
        super(BayesFlow, self).__init__()

        self.cINNs = [ConditionalInvertibleBlock(meta, theta_dim, alpha=alpha, permute=permute) for _ in
                      range(n_blocks)]
        self.summary_net = summary_net
        self.theta_dim = theta_dim

    def call(self, theta, x, inverse=False):
        """
        Performs one pass through an invertible chain (either inverse or forward).
        :param theta: tf.Tensor of shape (batch_size, inp_dim) -- the parameters theta ~ p(theta|x) of interest
        :param x: tf.Tensor of shape (batch_size, summary_dim) -- the conditional data x
        :param inverse: bool -- flag indicating whether to tun the chain forward or backwards
        :return:
        (z, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        x               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """
        if self.summary_net is not None:
            x = self.summary_net(x)
        if inverse:
            return self.inverse(theta, x)
        else:
            return self.forward(theta, x)

    def forward(self, theta, x):
        """
        Performs a forward pass through the chain.
        """
        z = theta
        log_det_Js = []
        for cINN in self.cINNs:
            z, log_det_J = cINN(z, x)
            log_det_Js.append(log_det_J)

        # Sum Jacobian determinants for all blocks to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return {'z': z, 'log_det_J': log_det_J}

    def inverse(self, z, x):
        """
        Performs a reverse pass through the chain.
        """
        theta = z
        for cINN in reversed(self.cINNs):
            theta = cINN(theta, x, inverse=True)
        return theta

    def sample(self, x, n_samples, to_numpy=False, training=False):
        """
        Samples from the inverse model given a single instance y or a batch of instances.
        :param x: tf.Tensor of shape (batch_size, summary_dim) -- the conditioning data of interest
        :param n_samples: int -- number of samples to obtain from the approximate posterior
        :param to_numpy: bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        :param training: bool -- flag used to indicate that samples are drawn in training time (BatchNorm)
        :return:
        theta_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, theta_dim)
        """
        # Summarize obs data if summary net available
        if self.summary_net is not None:
            x = self.summary_net(x, training=training)

        # In case x is a single instance
        if int(x.shape[0]) == 1:
            z_normal_samples = tf.random.normal(shape=(n_samples, self.theta_dim), dtype=tf.float32)
            theta_samples = self.inverse(z_normal_samples, tf.tile(x, [n_samples, 1]))
        # In case of a batch input, send a 3D tensor through the invertible chain and use tensordot
        # Warning: This tensor could get pretty big if sampling a lot of values for a lot of batch instances!
        else:
            z_normal_samples = tf.random.normal(shape=(n_samples, int(x.shape[0]), self.theta_dim), dtype=tf.float32)
            theta_samples = self.inverse(z_normal_samples, tf.stack([x] * n_samples))

        if to_numpy:
            return theta_samples.numpy()
        return theta_samples
