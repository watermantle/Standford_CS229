# Important note: you do not have to modify this file for your homework.

import util
import numpy as np


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))

    # grad regularized with l2
    #grad = -(1./m) * (X.T.dot(probs * Y) - 2 * theta)

    #regular grad
    grad = -(1. / m) * (X.T.dot(probs * Y))
    return grad

def logistic_regression(X, Y):
    """Train a logistic regression model"""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        # # learning rate decay
        # learning_rate /= i**2

        theta = theta - learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)

            break
    return


def main():
    # load dataset A and B
    #
    path = lambda data : "../data/ds1_" + data + ".csv"
    Xa, ya = util.load_csv(path("a"), add_intercept=True)
    Xb, yb = util.load_csv(path("b"), add_intercept=True)

    # print("===Start to train the model with dataset A===")
    # model_lr_A = logistic_regression(Xa, ya)
    print("===Start to train the model with dataset B===")
    model_lr_B = logistic_regression(Xb, yb)

    # plot training set A and B to investigate
    # Xa, ya = util.load_csv(path("a"), add_intercept=False)
    # Xb, yb = util.load_csv(path("b"), add_intercept=False)
    #
    # figa, axa = util.plot_points(Xa, ya)
    # figb, axb = util.plot_points(Xb, yb)
    #
    # figa.savefig("output/dsa.png")
    # figb.savefig("output/dsb.png")

    return

if __name__ == '__main__':
    main()
