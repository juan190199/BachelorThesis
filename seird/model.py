import numpy as np
import tensorflow as tf

EPSILON = 1e-4


def prior(n_samples, prior_bounds, parameter_names):
    """
    Calculate prior distributions using with suitable lower and upper bounds for each parameters.

    :param n_samples: int
        Number of samples to be drawn from the prior

    :param prior_bounds: ndarray of shape (2, n_parameters)
        Low and high boundaries for parameter priors

    :param parameter_names: list
        List of parameter names

    :return: ndarray of shape (n_samples, n_parameters)
        Sampled parameters from the prior
    """
    n_parameters = len(parameter_names)

    low = prior_bounds[0, :]
    high = prior_bounds[1, :]

    low_zip = zip(parameter_names, low)
    high_zip = zip(parameter_names, high)

    low_dict = dict(low_zip)
    high_dict = dict(high_zip)

    params = np.random.uniform(
        low=list(low_dict.values()),
        high=list(high_dict.values()),
        size=(n_samples, n_parameters)
    )

    return params


def version_prior(n_samples, version='v3', low_epsilon=None, up_epsilon=None, learning_noise=False):
    """

    :param n_samples: int
        Number of samples to be drawn from the prior

    :param version: string
        Version of the bachelor thesis

    :return: ndarray of shape (n_samples, n_parameters)
        Sampled parameters from the prior
    """
    if version == 'v1':
        parameter_names = ['beta', 'sigma', 'gamma', 'delta', 'rho']
        prior_bounds = np.array([[0.8, 0.45, 0.1, 0.01, 0.1], [2.25, 0.55, 1.0, 0.4, 0.6]])
    if version == 'v2':
        parameter_names = ['beta', 'sigma', 'gamma', 'delta', 'eta']
        prior_bounds = np.array([[0.8, 0.25, 0.1, 0.01, 0.025], [2.25, 0.75, 1.0, 0.4, 0.45]])
    if version == 'v3' or version == 'v4' or version == 'v6':
        parameter_names = ['beta', 'sigma', 'gamma', 'mu_I']
        prior_bounds = np.array([[0.8, 0.075, 0.01, 0.025], [2.25, 0.25, 0.4, 0.45]])
    if version == 'v5':
        parameter_names = ['beta', 'sigma', 'gamma', 'mu_I', 'epsilon']
        # if low_epsilon is not None and up_epsilon is not None:
        #     prior_bounds = np.array([[0.8, 0.075, 0.01, 0.025, low_epsilon], [2.25, 0.25, 0.4, 0.45, up_epsilon]])
        # else:
        #     prior_bounds = np.array([[0.8, 0.075, 0.01, 0.025, 0.05], [2.25, 0.25, 0.4, 0.45, 0.15]])

    params = prior(n_samples, prior_bounds=prior_bounds, parameter_names=parameter_names)

    if learning_noise:
        return params
    else:
        return params[:, :-1]


def version_data_model(parameters, t, initial_values, version='v3', learning_noise=False):
    """
    System of differential equations for SEIRD Model simulation

    :param parameters: ndarray of shape (n_parameters)


    :param t: ndarray of shape (tsteps, )
        Array with tsteps for tseries

    :param initial_values: ndarray of shape (5, )
        Initial values for SEIRD compartments

    :param version: string, default='v3'
        Version of the bachelor thesis

    :return: ndarray of shape (tsteps, 5)
        Tseries for each compartment
    """
    # SEIRD initial values
    S_0, E_0, I_0, R_0, D_0 = initial_values
    N = S_0 + E_0 + I_0 + R_0 + D_0
    S, E, I, R, D = [S_0], [E_0], [I_0], [R_0], [D_0]

    dt = t[1] - t[0]

    if version == 'v1':
        beta, sigma, gamma, delta, rho = parameters
        for _ in t[1:]:
            next_S = S[-1] - ((beta * S[-1] * I[-1]) / N) * dt
            next_E = E[-1] + (beta * S[-1] * I[-1] / N - delta * E[-1]) * dt
            next_I = I[-1] + (delta * E[-1] - (1 - gamma) * sigma * I[-1] - gamma * rho * I[-1]) * dt
            next_R = R[-1] + ((1 - gamma) * sigma * I[-1]) * dt
            next_D = D[-1] + gamma * rho * I[-1] * dt

            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            D.append(next_D)

    if version == 'v2':
        beta, sigma, gamma, delta, eta = parameters
        for _ in t[1:]:
            next_S = S[-1] - ((beta * S[-1] * I[-1]) / N) * dt
            next_E = E[-1] + (beta * S[-1] * I[-1] / N - delta * E[-1]) * dt
            next_I = I[-1] + (delta * E[-1] - (1 - gamma) * sigma * I[-1] - eta * I[-1]) * dt
            next_R = R[-1] + ((1 - gamma) * sigma * I[-1]) * dt
            next_D = D[-1] + eta * I[-1] * dt

            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            D.append(next_D)

    if version == 'v3' or version == 'v4' or version == 'v6':
        beta, sigma, gamma, mu_I = parameters
        for _ in t[1:]:
            next_S = S[-1] - ((beta * S[-1] * I[-1]) / N) * dt
            next_E = E[-1] + (beta * S[-1] * I[-1] / N - sigma * E[-1]) * dt
            next_I = I[-1] + (sigma * E[-1] - gamma * I[-1] - mu_I * I[-1]) * dt
            next_R = R[-1] + (gamma * I[-1]) * dt
            next_D = D[-1] + mu_I * I[-1] * dt

            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            D.append(next_D)

    if version == 'v5':
        if learning_noise:
            beta, sigma, gamma, mu_I, epsilon = parameters
        else:
            beta, sigma, gamma, mu_I = parameters

        for _ in t[1:]:
            next_S = S[-1] - ((beta * S[-1] * I[-1]) / N) * dt
            next_E = E[-1] + (beta * S[-1] * I[-1] / N - sigma * E[-1]) * dt
            next_I = I[-1] + (sigma * E[-1] - gamma * I[-1] - mu_I * I[-1]) * dt
            next_R = R[-1] + (gamma * I[-1]) * dt
            next_D = D[-1] + mu_I * I[-1] * dt

            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            R.append(next_R)
            D.append(next_D)

    return np.stack([S, E, I, R, D]).T


def data_generator(n_samples, T=100, dt=1, N=1000, version='v4', S=False, E=False, low_epsilon=None, up_epsilon=None, learning_noise=False, to_tensor=False):
    """

    :param n_samples: int
        Number of SEIRD Model tseries to be generated

    :param T: int
        Maximum number of tsteps

    :param dt: int
        Tsteps difference

    :param N: int
        Total number of population

    :param version: string, default='v3'
        Version type

    :param to_tensor: boolean, default=True
        Boolean variable, if True convert X and params ndarrays to tensor

    :return: dict
        Dictionary with keys: 'X' (SEIRD Model tseries) and 'params' (Parameters)
    """

    # Time steps
    t = np.linspace(0, T, int(T / dt))

    # SEIRD initial values
    initial_values = 1 - 1 / N, 1 / N, 0., 0., 0.

    # Sample parameters from the prior distributions.
    # The parameters are sampled from the prior distribution for each instance data model
    params = version_prior(n_samples=n_samples, version=version, low_epsilon=low_epsilon, up_epsilon=up_epsilon, learning_noise=learning_noise)

    # Generate tseries
    X = np.apply_along_axis(
        func1d=version_data_model,
        axis=1,
        arr=params,
        t=t,
        initial_values=initial_values,
        version=version,
        learning_noise=learning_noise
    )

    if version == 'v5':
        noisy_X = X.copy()

        for i in range(noisy_X.shape[0]):
            noise = np.random.lognormal(mean=0, sigma=params[i, -1], size=noisy_X.shape[1:])
            noisy_X[i] = noisy_X[i] * noise

        # Sanity checks
        noisy_X[noisy_X < EPSILON] = EPSILON
        noisy_X[noisy_X > 1] = 1 - EPSILON

        if to_tensor:
            params = tf.convert_to_tensor(params, dtype=tf.float32)
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            noisy_X = tf.convert_to_tensor(noisy_X, dtype=tf.float32)

        return {'params': params, 'X': X, 'noisy_X': noisy_X}

    elif version == 'v6':
        if S and E:
            # Drop S, E compartment tseries
            dropped_X = X[:, :, 2:].copy()
        elif E:
            # Drop E compartment tseries
            dropped_X = np.concatenate((X[:, :, :1], X[:, :, 2:]), axis=2)
        elif S:
            # Drop S compartment tseries
            dropped_X = X[:, :, 1:].copy()

        if to_tensor:
            params = tf.convert_to_tensor(params, dtype=tf.float32)
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        return {'params': params, 'X': X, 'dropped_X': dropped_X}

    else:
        if to_tensor:
            params = tf.convert_to_tensor(params, dtype=tf.float32)
            X = tf.convert_to_tensor(X, dtype=tf.float32)

        return {'params': params, 'X': X}



