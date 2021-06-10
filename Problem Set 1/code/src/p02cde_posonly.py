import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***

    #####################################################
    # Problem(c)
    #####################################################

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    logreg_c = LogisticRegression()
    logreg_c.fit(x_train, t_train)
    t_pred = logreg_c.predict(x_test)
    util.plot(x=x_test, y=t_test, theta=logreg_c.theta, save_path='output/p02c{0}.png'.format(pred_path_c[16:-4]))
    np.savetxt(pred_path_c, t_pred, fmt='%d')

    #####################################################
    # Problem(d)
    #####################################################
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    logreg_d = LogisticRegression()
    logreg_d.fit(x_train, y_train)
    y_pred = logreg_c.predict(x_test)
    util.plot(x=x_test, y=y_test, theta=logreg_d.theta, save_path='output/p02d{0}.png'.format(pred_path_c[16:-4]))
    np.savetxt(pred_path_d, y_pred, fmt='%d')

    #####################################################
    # Problem(e)
    #####################################################
    x_val, y_val = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    y_p = logreg_d.predict(x_val, p=True)
    alpha = y_p.mean()
    correction = 1 + np.log(2 / alpha -1) / logreg_d.theta[0]

    # Plot corrected decision boundary
    util.plot(x=x_test, y=t_test, theta=logreg_d.theta, save_path='output/p02d{0}.png', correction=correction)

    p_corrected = y_p / alpha
    t_pred_corrected = np.where(p_corrected >= 0.5, 1, 0)
    np.savetxt(pred_path_e, t_pred_corrected, fmt='%d')

    # *** END CODER HERE
