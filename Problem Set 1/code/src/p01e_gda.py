import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    GDA_model = GDA()
    GDA_model.fit(x_train, y_train)

    y_pred = GDA_model.predict(x_eval)
    # plot decision boundary
    util.plot(x=x_train, y=y_train, theta=GDA_model.theta, save_path='output/p01e{0}.png'.format(pred_path[-5]),
              x_eval=x_eval, y_pred=y_pred)
    np.savetxt(pred_path, y_pred, fmt='%d')
    # *** START CODE HERE ***

    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape

        ### calculate GDA parameters
        num_y1 = y[y == 1].sum()
        phi = num_y1.sum() / m

        mu0 = x[y == 0].sum(axis=0) / (m - num_y1)
        mu1 = x[y == 1].sum(axis=0) / num_y1
        sigma = (((x[y == 0] - mu0).T).dot(x[y == 0] - mu0) + ((x[y == 1] - mu1).T).dot(x[y == 1] - mu1)) / m

        ### theta calculation based on GDA parameters
        theta = np.linalg.inv(sigma).dot(mu1 - mu0)
        theta0 = -0.5 * (mu1 - mu0).dot(np.linalg.inv(sigma)).dot(mu1 + mu0) - np.log((1 - phi) / phi)
        self.theta = np.hstack([theta0, theta])

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        h_x_opt = util.sigmoid(x.dot(self.theta))
        return np.where(h_x_opt >= 0.5, 1, 0)
        # *** END CODE HERE
