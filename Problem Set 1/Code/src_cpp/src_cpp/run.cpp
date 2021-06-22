
#include <iostream>
#include <armadillo>
#include "util.hpp"
#include "linear_model.hpp"
#include "p01b_logreg.hpp"



int main() {
	std::string root, path1_train, path1_eval;
	root = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\data\\";
	path1_train = "ds1_train.csv";
	path1_eval = "ds1_valid.csv";

	tuple<mat, mat> data_train = util::load_dataset(root + path1_train);
	tuple<mat, mat> data_eval = util::load_dataset(root + path1_eval);
	mat X_train, y_train, X_eval, y_eval;

	X_train = get<0>(data_train);
	y_train = get<1>(data_train);

	X_eval = get<0>(data_eval);
	y_eval = get<1>(data_eval);

	LogisticRegression logreg;
	logreg.fit(X_train, y_train);
	
	mat y_pred;
	y_pred = logreg.predict(X_eval);
	
	cout << y_pred << endl;

	return 0;
}
