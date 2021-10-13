import numpy as np

from seird.model import data_generator


def data_model_sc(X, params, T=100, to_tensor=True):
    """
    Sanity checks for SEIRD Model tseries

    :param X: ndarray of shape (n_samples, n_tsteps, 5)
        SEIRD Model tseries

    :param params: ndarray of shape (n_samples, n_parameters)
        Parameters

    :param T: int
        Maximum number of tsteps

    :param to_tensor: boolean
        Boolean variable to convert ndarray data to tensor

    :return:
        X: ndarray of shape (n_samples, n_tsteps, 5)
            SEIRD Model tseries

        params: ndarray of shape (n_samples, n_parameters)
            Parameters
    """
    N = X.shape[0]
    n_instances = 0

    # Limit superior of population that continues susceptible at the last time step
    threshold_S = 0.25

    # Ensure n_samples
    while n_instances != N:
        if N - n_instances != N:
            extra_data = data_generator(n_samples=(N - n_instances), T=T, dt=dt, N=N, to_tensor=to_tensor)
            extra_X = np.array(extra_data['X'])
            extra_params = np.array(extra_data['params'])

            X = np.concatenate((X, extra_X))
            params = np.concatenate((params, extra_params))

        # Sanity check

