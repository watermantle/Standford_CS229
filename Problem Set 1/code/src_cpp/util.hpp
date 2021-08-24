/*
header file to store some reusable functions
*/
#include <iostream>
#include <fstream>
#include <armadillo>


using namespace std;
using namespace arma;

#ifndef util_HHP
#define util_HHP

namespace util {
	// load dataset from a csv file
	tuple<mat, mat> load_dataset(std::string path, int label_loc = 2, int n_inputs = 2, bool ifadd_intercept = true);
	// add ones to a matrix
	mat add_intercept(mat x);
	// sigmoid function
	mat sigmoid(mat z);
	// mean squared errors calculation
	double MSE(vec& y_pred, vec& y_true);
};

#endif // end