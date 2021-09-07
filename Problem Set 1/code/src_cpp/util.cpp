#include "util.hpp"
#include <iostream>
#include <fstream>
#include <armadillo>

namespace util {
	mat add_intercept(mat& x) {
		/*
		Args:
			x: 2D matrix

		Return:
			New matrix same as x with 1's in the 0th column
		*/
		mat new_x(x.n_rows, x.n_cols + 1);
		new_x.col(0) = ones(x.n_rows);
		new_x.cols(1, new_x.n_cols - 1) = x;
		return new_x;
	}

	double MSE(vec& y_pred, vec& y_true) {
		arma::uword m = y_pred.n_rows;
		return sum(pow((y_pred - y_true), 2)) / m;
	}

	tuple<mat, mat> load_dataset(std::string path, int label_loc, int n_inputs, bool ifadd_intercept) {
		/*load dataset from a CSV file
		 Args:
			 csv_path: Path to CSV file containing dataset.
			 label_col: Name of column to use as labels should be an int.
			 add_intercept: Add an intercept entry to x-values.
			 n_inputs: number of the inputs with defult value of 2

		Returns:
			a tulpe contains inputs (Xs) and label (ys)
		*/

		// read csv file, drop headers
		mat data;
		data.load(path, arma::csv_ascii);
		data = data.submat(1, 0, data.n_rows - 1, data.n_cols - 1);

		// extract needed inputs & label
		int inputs_col = n_inputs - 1;
		mat inputs = data.submat(0, 0, data.n_rows - 1, inputs_col);
		mat label = data.col(label_loc);

		if (ifadd_intercept) { inputs = add_intercept(inputs); };
		return make_tuple(inputs, label);
	}

	// sigmoid function to computer logsitic hypothetical function
	mat sigmoid(mat& z) {
		return 1 / (1 + exp(-z));
	}
}
