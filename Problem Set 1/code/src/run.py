import argparse
from p01b_logreg import main as p01b

parser = argparse.ArgumentParser()
parser.add_argument('p_num', nargs='?', type=int, default=0,
                    help='Problem number to run, 0 for all problems.')
args = parser.parse_args()

p01b(train_path='../data/ds1_train.csv', eval_path='../data/ds1_valid.csv',
     pred_path='output/p01b_pred_1.txt')