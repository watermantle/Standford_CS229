#include <iostream>
#include <fstream>
#include <armadillo>


using namespace std;
using namespace arma;


tuple<mat, mat> load_dataset(std::string path, int lable=1, bool add_intercept=false);
mat add_intercept(mat x);

mat add_intercept(mat x) {
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

tuple<mat, mat> load_dataset(std::string path, int lable, bool add_intercept) {
	/*load dataset from a CSV file
	 Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels should be an int.
         add_intercept: Add an intercept entry to x-values.

    Returns:
		a tulpe contains inputs (Xs) and label (ys)
	*/
	return make_tuple(zeros(1), ones(1));
}

int main() {
	
}