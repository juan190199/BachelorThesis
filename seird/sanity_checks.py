import numpy as np


def sampling_sc(*args, version='v4', learning_noise=False):
    """
    Ensures sampled means of the sampled posteriors for each time series obey boundaries prior distributions

    :param args:
    :return:
    """

    if version in ['v1']:
        parameter_names = ['beta', 'sigma', 'gamma', 'delta', 'rho']
        lower_bound = [0.8, 0.45, 0.1, 0.01, 0.1]
        upper_bound = [2.25, 0.55, 1.0, 0.4, 0.6]
    if version in ['v1_1']:
        parameter_names = ['beta', 'sigma', 'gamma', 'delta', 'eta']
        lower_bound = [0.8, 0.25, 0.1, 0.01, 0.025]
        upper_bound = [2.25, 0.75, 1.0, 0.4, 0.45]
    if version in ['v2', 'v3', 'v5']:
        parameter_names = ['beta', 'sigma', 'gamma', 'mu_I']
        lower_bound = [0.8, 0.075, 0.01, 0.025]
        upper_bound = [2.25, 0.25, 0.4, 0.45]
    if version in ['v4', 'v5_2']:
        if learning_noise:
            parameter_names = ['beta', 'sigma', 'gamma', 'mu_I', 'epsilon']
            lower_bound = [0.8, 0.075, 0.01, 0.025, 0.05]
            upper_bound = [2.25, 0.25, 0.4, 0.45, 0.15]
        else:
            parameter_names = ['beta', 'sigma', 'gamma', 'mu_I']
            lower_bound = [0.8, 0.075, 0.01, 0.025]
            upper_bound = [2.25, 0.25, 0.4, 0.45]
    if version in ['v6']:
        parameter_names = [
            'beta', 'sigma', 'gamma', 'xi', 'mu_I',
            'beta_Q', 'sigma_Q', 'gamma_Q', 'mu_Q',
            'theta_E', 'theta_I', 'psi_E', 'psi_I',
            'nu', 'mu_0', 'q'
        ]
        lower_bound = [
            0.3, 0.075, 0.025, 0.0005, 0.025,
            0.15, 0.075, 0.025, 0.025,
            0.01, 0.01, 1.0, 1.0,
            0.0, 0.0, 0.0
        ]
        upper_bound = [
            1.0, 0.35, 0.075, 0.015, 0.075,
            0.5, 0.35, 0.075, 0.075,
            0.07, 0.07, 1.0, 1.0,
            0.0, 0.0, 0.0
        ]

    if len(args) == 2:
        params_samples_mean, params_test = args
        params_samples = None
        X_test = None
    elif len(args) == 4:
        params_samples_mean, params_test, params_samples, X_test = args
    elif len(args) == 5:
        params_samples_mean, params_test, params_samples, noisy_X_test, X_test = args
    elif len(args) == 6:
        params_samples_mean, params_test, params_samples, noisy_dropped_X_test, X_test, noisy_X_test = args

        # Mask per parameter (Check if sampled parameters are bigger than lower bound and smaller than upper bound)
    mask = [
        (params_samples_mean[:, i] > lower_bound[i]) * (params_samples_mean[:, i] < upper_bound[i]) for i in
        range(params_samples_mean.shape[1])
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

    if len(args) == 6:
        sc_params_samples = np.delete(params_samples, np.argwhere(mask == False)[:, 0], axis=1)
        sc_noisy_dropped_X_test = np.delete(noisy_dropped_X_test, np.argwhere(mask == False)[:, 0], axis=0)
        sc_X_test = np.delete(X_test, np.argwhere(mask == False)[:, 0], axis=0)
        sc_noisy_X_test = np.delete(noisy_X_test, np.argwhere(mask == False)[:, 0], axis=0)

        assert sc_X_test.shape[0] == sc_params_samples_means.shape[0]
        return sc_params_samples_means, sc_params_test, sc_params_samples, sc_noisy_dropped_X_test, sc_X_test, sc_noisy_X_test
