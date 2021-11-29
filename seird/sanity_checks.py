import numpy as np

from seird.model import data_generator


# def data_model_sc(X, params, T=100, dt=1, to_tensor=True):
#     """
#     Sanity checks for SEIRD Model tseries
#
#     :param X: ndarray of shape (n_samples, n_tsteps, 5)
#         SEIRD Model tseries
#
#     :param params: ndarray of shape (n_samples, n_parameters)
#         Parameters
#
#     :param T: int
#         Maximum number of tsteps
#
#     :param to_tensor: boolean
#         Boolean variable to convert ndarray data to tensor
#
#     :return:
#         X: ndarray of shape (n_samples, n_tsteps, 5)
#             SEIRD Model tseries
#
#         params: ndarray of shape (n_samples, n_parameters)
#             Parameters
#     """
#     N = X.shape[0]
#     n_instances = 0
#
#     # Limit superior of population that continues susceptible at the last time step
#     threshold_S = 0.25
#
#     # Ensure n_samples
#     while n_instances != N:
#         if N - n_instances != N:
#             extra_data = data_generator(n_samples=(N - n_instances), T=T, dt=dt, N=N, to_tensor=to_tensor)
#             extra_X = np.array(extra_data['X'])
#             extra_params = np.array(extra_data['params'])
#
#             X = np.concatenate((X, extra_X))
#             params = np.concatenate((params, extra_params))


def sampling_sc(*args, version='v4'):
    """
    Ensures sampled means of the sampled posteriors for each time series obey boundaries prior distributions

    :param args:
    :return:
    """

    if version == 'v1':
        parameter_names = ['beta', 'sigma', 'gamma', 'delta', 'rho']
        lower_bound = [0.8, 0.45, 0.1, 0.01, 0.1]
        upper_bound = [2.25, 0.55, 1.0, 0.4, 0.6]
    if version == 'v2':
        parameter_names = ['beta', 'sigma', 'gamma', 'delta', 'eta']
        lower_bound = [0.8, 0.25, 0.1, 0.01, 0.025]
        upper_bound = [2.25, 0.75, 1.0, 0.4, 0.45]
    if version == 'v3' or version == 'v4':
        parameter_names = ['beta', 'sigma', 'gamma', 'mu_I']
        lower_bound = [0.8, 0.075, 0.01, 0.025]
        upper_bound = [2.25, 0.25, 0.4, 0.45]

        # if False:
        #     params_samples = args[0]
        #
        #     # Mask per parameter
        #     mask = [
        #         (params_samples[:, i] > lower_bound[i]) * (params_samples[:, i] < upper_bound[i])
        #         for i in range(len(lower_bound))
        #     ]
        #     mask = np.array(mask).T
        #
        #     sc_paarams_samples = np.delete(params_samples, np.argwhere(mask == False)[:, 0], axis=0)
        #     return sc_paarams_samples

        # else:
    if len(args) == 2:
        params_samples_mean, params_test = args
        params_samples = None
        X_test = None
    elif len(args) == 4:
        params_samples_mean, params_test, params_samples, X_test = args
    elif len(args) == 5:
        params_samples_mean, params_test, params_samples, noisy_X_test, X_test = args

        # Mask per parameter (Check if sampled parameters are bigger than lower bound and smaller than upper bound)
    mask = [
        (params_samples_mean[:, i] > lower_bound[i]) * (params_samples_mean[:, i] < upper_bound[i]) for i in
        range(len(lower_bound))
    ]
    mask = np.array(mask).T

    sc_params_samples_means = np.delete(params_samples_mean, np.argwhere(mask == False)[:, 0], axis=0)
    sc_params_test = np.delete(params_test, np.argwhere(mask == False)[:, 0], axis=0)

    if len(args) == 4:
        sc_params_samples = np.delete(params_samples, np.argwhere(mask == False)[:, 0], axis=1)
        sc_X_test = np.delete(X_test, np.argwhere(mask == False)[:, 0], axis=0)

        assert sc_X_test.shape[0] == sc_params_samples_means.shape[0]
        return sc_params_samples_means, sc_params_test, sc_params_samples, sc_X_test

    if len(args) == 5:
        sc_params_samples = np.delete(params_samples, np.argwhere(mask == False)[:, 0], axis=1)
        sc_noisy_X_test = np.delete(noisy_X_test, np.argwhere(mask == False)[:, 0], axis=0)
        sc_X_test = np.delete(X_test, np.argwhere(mask == False)[:, 0], axis=0)

        assert sc_X_test.shape[0] == sc_params_samples_means.shape[0]
        return sc_params_samples_means, sc_params_test, sc_params_samples, sc_noisy_X_test, sc_X_test
