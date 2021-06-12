import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    model_LWR = LocallyWeightedLinearRegression(tau=tau)
    model_LWR.fit(x_train, y_train)
    y_pred = model_LWR.predict(x_val)

    # mean squared error
    mse = ((y_pred - y_val) ** 2).mean()
    fig, ax = util.plt.subplots()
    ax.plot(x_train[::, -1], y_train, 'bx')
    ax.plot(x_val[::, -1], y_pred, 'ro')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("MSE: {:.4f}".format(mse))
    fig.savefig('output/p05b.png')

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        # No calculation, only set up variable to make them available in predict method
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # weights matrix, a diagonal matrix
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            W = np.diag(np.exp(-np.linalg.norm(x[i] - self.x, ord=2, axis=1) / (2 * (self.tau ** 2))))
            self.theta = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(self.x.T).dot(W).dot(self.y)
            y_pred[i] = x[i].dot(self.theta)
        return y_pred
        # *** END CODE HERE ***
