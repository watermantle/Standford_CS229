import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    ## train the model
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    y_pred = logreg.predict(x_eval)
    # plot decision boundary
    util.plot(x=x_train, y=y_train, theta=logreg.theta, save_path='output/p01b{0}.png'.format(pred_path[16:-4]),
              x_eval=x_eval, y_pred=y_pred)
    np.savetxt(pred_path, y_pred, fmt='%d')
    # *** END CODE HERE ***

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        
        ## init theta with Shape of (n, )
        m, n = x.shape
        self.theta = np.zeros(n)

        ## apply newton's method
        n_iter = 0
        while n_iter < self.max_iter:

            h_x = util.sigmoid(x.dot(self.theta))
            gradient_J = -x.T.dot(y - h_x) / m
            H_J = (x.T * h_x * (1- h_x)).dot(x) / m

            ## update theta & check if converge
            step = np.linalg.inv(H_J).dot(gradient_J)
            self.theta -= step
            if np.linalg.norm(step, ord=1) < self.eps: break
            n_iter += 1
        
        # *** END CODE HERE ***

    def predict(self, x, p=False):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).
            p: if return prob of the output

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        h_x_opt = util.sigmoid(x.dot(self.theta))
        if p:
            return h_x_opt
        else:
            return np.where(h_x_opt >= 0.5, 1, 0)
        # *** END CODE HERE ***
