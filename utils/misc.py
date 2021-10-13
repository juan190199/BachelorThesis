import numpy as np
from sklearn.metrics import r2_score

from itertools import combinations

import matplotlib.pyplot as plt


def true_vs_estimated(model, X_test, theta_test, n_samples, param_names,
                          figsize=(20, 4), params_samples_mean=None, show=True, filename=None, font_size=12):
    """
    Scatter plot  of the estimated posterior means vs true values.

    :param model:

    :param X_test: ndarray of shape (n_test, n_ts, n_compartments)
        Test data

    :param theta_test: ndarray of shape (n_test, n_parameters)
        Ground truth

    :param n_samples: int
        Number of posterior samples for each time serie

    :param param_names: list
        List with parameter names

    :param figsize: tuple
        Figure size

    :param params_samples_mean:

    :param show: boolean

    :param filename: string
        Save the file under the filename

    :param font_size: float
        Figure font size

    :return:
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Convert true parameters to numpy
    # theta_test = theta_test.numpy()

    # Determine figure layout
    if len(param_names) >= 6:
        n_col = int(np.ceil(len(param_names) / 2))
        n_row = 2
    else:
        n_col = int(len(param_names))
        n_row = 1

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Initialize posterior means matrix, if none specified
    if params_samples_mean is None:
        params_samples_mean = model.sample(X_test, n_samples, to_numpy=True).mean(axis=0)

    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        # Plot analytic vs estimated
        axarr[j].scatter(params_samples_mean[:, j], theta_test[:, j], color='black', alpha=0.4)

        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])

        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')

        # Compute NRMSE
        rmse = np.sqrt(np.mean((params_samples_mean[:, j] - theta_test[:, j]) ** 2))
        nrmse = rmse / (theta_test[:, j].max() - theta_test[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=axarr[j].transAxes,
                      size=10)

        # Compute R2
        r2 = r2_score(theta_test[:, j], params_samples_mean[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                      horizontalalignment='left',
                      verticalalignment='center',
                      transform=axarr[j].transAxes,
                      size=10)

        # Label plot x-axis
        axarr[j].set_xlabel('Estimated')

        # Label plot
        axarr[j].set_ylabel('True')
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)

    # Adjust spaces
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot title
    f.suptitle("True vs Estimated")

    if show:
        plt.show()

    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_scatter.png".format(filename, X_test.shape[1]), dpi=600)


def plot_parameters_correlation(parameters, parameter_names, figsize=(20, 10), show=True, font_size=11):
    """

    :param parameters:
    :param parameter_names:
    :param figsize:
    :param show:
    :param filename:
    :param font_size:
    :return:
    """
    plt.clf()
    # Plot settings
    plt.rcParams['font.size'] = font_size

    n_parameters = len(parameter_names)

    # Get all possible combinations between parameters
    comp_list = list(combinations(np.arange(parameters.shape[1]), 2))
    # List of comp_params
    comp_params = []
    comp_param_names = []
    for (i, j) in comp_list:
        comp_params.append(np.column_stack((parameters[:, i], parameters[:, j])))
        comp_param_names.append([parameter_names[i], parameter_names[j]])

    fig, ax = plt.subplots(2, int(len(comp_list) / 2), figsize=figsize)
    for idx in range(int(len(comp_list) / 2)):
        ax[0, idx].scatter(comp_params[idx][:, 0], comp_params[idx][:, 1], color='black', alpha=0.4)
        ax[0, idx].set_xlabel(comp_param_names[idx][0], fontsize=15)
        ax[0, idx].set_ylabel(comp_param_names[idx][1], fontsize=15)

        ax[1, idx].scatter(comp_params[int(len(comp_list) / 2) + idx][:, 0], comp_params[int(len(comp_list) / 2) + idx][:, 1], color='black', alpha=0.4)
        ax[1, idx].set_xlabel(comp_param_names[idx + int(len(comp_list) / 2)][0], fontsize=15)
        ax[1, idx].set_ylabel(comp_param_names[idx + int(len(comp_list) / 2)][1], fontsize=15)

    fig.suptitle('Parameter correlation', fontsize=20)
    # Adjust spaces
    plt.tight_layout()

    if show:
        plt.figure(figsize=figsize)
        plt.show()