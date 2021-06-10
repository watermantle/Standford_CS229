import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    # *** START CODE HERE ***
    pois_reg = PoissonRegression(step_size=lr)
    pois_reg.fit(x_train, y_train)

    y_pred = pois_reg.predict(x_eval)

    # Plot results compared to the true result
    fig, ax = util.plt.subplots()
    # Index of the points that are error is less than 10%
    qf_idx = (np.linalg.norm(y_pred - y_eval, ord=1) / y_eval) < 0.1
    ax.plot(y_eval[qf_idx], y_pred[qf_idx], 'bx')
    ax.plot(y_eval[~qf_idx], y_pred[~qf_idx], 'rx')
    np.savetxt(pred_path, y_pred, fmt='%d')
    ax.set_xlabel('true counts')
    ax.set_ylabel('predict counts')
    fig.savefig('output/p03d.png')
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # initiate theta with zeros

        m, n = x.shape
        self.theta = np.zeros(n)

        while True:
            # pick up a random index to perform mini batch gradient ascent with size of 100
            batch_size = 100
            idx_rnd = np.random.choice(range(m), size=batch_size)
            xi = x[idx_rnd]
            yi = y[idx_rnd]
            gradient = xi.T.dot(yi - np.exp(xi.dot(self.theta))) / batch_size

            self.theta += self.step_size * gradient
            if np.linalg.norm(self.step_size * gradient, ord=1) < self.eps: break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***
