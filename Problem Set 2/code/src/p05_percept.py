import math

import matplotlib.pyplot as plt
import numpy as np

import util

def initial_state():
    """Return the initial state for the perceptron.

    This function computes and then returns the initial state of the perceptron.
    Feel free to use any data type (dicts, lists, tuples, or custom classes) to 
    contain the state of the perceptron.

    The state will store coefficient C, the x_(i-1) x, and previous sum sum_pre (to lower computational cost)
    """

    # *** START CODE HERE ***
    return []
    # *** END CODE HERE ***


def predict(state, kernel, x_i):
    """Perform a prediction on a given instance x_i given the current state and the kernel.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        x_i: A vector containing the features for a single instance
    
    Returns:
        Returns the prediction (i.e 0 or 1)
    """
    # *** START CODE HERE ***
    # accumulated sum of C * K(x_j, x_(i+1))
    sum_acc = 0
    for C, x in state:
        sum_acc += C * kernel(x, x_i)

    return sign(sum_acc)
    # *** END CODE HERE ***


def update_state(state, kernel, learning_rate, x_i, y_i):
    """Updates the state of the perceptron.

    Args:
        state: The state returned from initial_state()
        kernel: A binary function that takes two vectors as input and returns the result of a kernel
        learning_rate: The learning rate for the update
        x_i: A vector containing the features for a single instance
        y_i: A 0 or 1 indicating the label for a single instance
    """
    # *** START CODE HERE ***
    # compute constant C and append a scalar(sum_cur) to the state
    C = learning_rate * (y_i - predict(state, kernel, x_i))

    state.append((C, x_i))
    # *** END CODE HERE ***

def sign(a):
    """Get the sign of a scalar input."""
    return np.where(a >=0, 1, 0).item()

def dot_kernel(a, b):
    """An implementation of a dot product kernel.

    Args:
        a: A vector
        b: A vector
    """
    return np.dot(a, b)

def rbf_kernel(a, b, sigma=1):
    """An implementation of the radial basis function kernel.

    Args:
        a: A vector
        b: A vector
        sigma: The radius of the kernel
    """

    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)

def train_perceptron(kernel_name, kernel, learning_rate):
    """Train a perceptron with the given kernel.

    This function trains a perceptron with a given kernel and then uses that perceptron to make predictions.
    The output predictions are saved to src/output/p05_{kernel_name}_predictions.txt
    The output plots are saved to src/output_{kernel_name}_output.pdf

    Args:
        kernel_name: The name of the kernel
        kernel: The kernel function
        learning_rate: The learning rate for training
    """
    train_x, train_y = util.load_csv('../data/ds5_train.csv')
    test_x, test_y = util.load_csv('../data/ds5_test.csv')

    state = initial_state()

    # go over the training set
    for xi, yi in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, xi, yi)

    fig, ax = util.plot_points(test_x, test_y)
    util.plot_contour(lambda a: predict(state, kernel, a), ax)
    ax.set_title("{0} kernel with learning rate of {1}".format(kernel_name, learning_rate))
    fig.savefig('./output/p05_{}_output.png'.format(kernel_name))


    predict_y = [predict(state, kernel, test_x[i, :]) for i in range(test_y.shape[0])]

    np.savetxt('./output/p05_{}_predictions.csv'.format(kernel_name), predict_y)

def main():
    train_perceptron('dot', dot_kernel, 0.5)
    train_perceptron('rbf', rbf_kernel, 0.5)

if __name__ == "__main__":
    main()
