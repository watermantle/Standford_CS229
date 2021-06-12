import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    # *** START CODE HERE ***
    # initiate a model
    model_LWR = LocallyWeightedLinearRegression(tau=0.5)
    model_LWR.fit(x_train, y_train)
    # initiate plot
    n_tau = len(tau_values)
    n_cols = 3
    n_rows = round(n_tau / 3)

    figs, axes = util.plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 9),
                                  gridspec_kw={'wspace':0.05, 'hspace':0.2}, subplot_kw={'xticks':[], 'yticks':[]})

    # a list of mse to store processing params
    MSEs = np.array([])

    # train model + plot
    for i in range(n_tau):
        tau = tau_values[i]
        model_LWR.tau = tau
        ax = axes.flat[i]
        try:
            y_pred = model_LWR.predict(x_val)
            mse = util.MSE(y_pred, y_val)
            MSEs = np.append(MSEs, mse)

            ax.plot(x_train[::, -1], y_train, 'bx')
            ax.plot(x_val[::, -1], y_pred, 'ro')
            title = "tau: {} and MSE: {:.4f}".format(tau, mse)
            ax.set_title(title)
        except:
            text = "tau={} is not available".format(tau)
            ax.text(0.05, 0.4, text, fontsize=15)

    # plot the best outcome
    best_tau = tau_values[MSEs.argmin()]
    model_LWR.tau = best_tau
    model_LWR.fit(x_train, y_train)
    y_pred = model_LWR.predict(x_test)
    mse = util.MSE(y_pred, y_test)
    title = "tau: {} and MSE: {:.4f}".format(best_tau, mse)
    fig, ax = util.plt.subplots()
    ax.plot(x_train[::, -1], y_train, 'bx')
    ax.plot(x_test[::, -1], y_pred, 'ro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)

    # save outcomes
    fig.savefig('output/p05c_best.png')
    figs.savefig('output/p05c_aggregated.png')
    np.savetxt(pred_path, y_pred, fmt='%d')
    # *** END CODE HERE ***
