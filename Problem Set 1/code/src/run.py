import argparse
from p01b_logreg import main as p01b
from p01e_gda import main as p01e
from p02cde_posonly import main as p02
from p03d_poisson import main as p03
from p05b_lwr import main as p05b
from p05c_tau import main as p05c

parser = argparse.ArgumentParser()
parser.add_argument('p_num', nargs='?', type=int, default=0,
                    help='Problem number to run, 0 for all problems.')
args = parser.parse_args()

# p01b(train_path='../data/ds1_train.csv', eval_path='../data/ds1_valid.csv',
#      pred_path='output/p01b_pred_logreg_ds1.csv')
# p01b(train_path='../data/ds2_train.csv', eval_path='../data/ds2_valid.csv',
#      pred_path='output/p01b_pred_logreg_ds2.csv')
#
# p01e(train_path='../data/ds1_train.csv', eval_path='../data/ds1_valid.csv',
#      pred_path='output/p01e_pred_GDA_ds1.csv')
# p01e(train_path='../data/ds2_train.csv', eval_path='../data/ds2_valid.csv',
#      pred_path='output/p01e_pred_GDA_ds2.csv')

# p02(train_path='../data/ds3_train.csv', valid_path='../data/ds3_valid.csv',
#     test_path='../data/ds3_test.csv', pred_path='output/p02X_pred.csv')
#
# p03(lr=1e-7, train_path='../data/ds4_train.csv', eval_path='../data/ds4_valid.csv',
#     pred_path='output/p03d_pred.csv')

# p05b(tau=0.5, train_path='../data/ds5_train.csv',
#      eval_path='../data/ds5_valid.csv')

p05c(tau_values=[1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 5e-1, 1e0, 3e0, 5e0],
     train_path='../data/ds5_train.csv', valid_path='../data/ds5_valid.csv',
     test_path='../data/ds5_test.csv', pred_path='output/p05c_pred.csv')