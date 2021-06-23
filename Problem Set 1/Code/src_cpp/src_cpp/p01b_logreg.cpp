/*
source file to apply logistic regression
*/
#include <armadillo>
#include "p01b_logreg.hpp"
#include "util.hpp"

using namespace std;
using namespace arma;

// constructors
LogisticRegression::LogisticRegression() : LinearModel(){};
LogisticRegression::LogisticRegression(const LogisticRegression& source) : LinearModel(source) {};
LogisticRegression::LogisticRegression(mat& theta, const double& step_size, const int& max_iter, const double& eps) : LinearModel(theta, step_size, max_iter, eps) {};
LogisticRegression::~LogisticRegression() {};

// assignment operator
LogisticRegression& LogisticRegression::operator=(const LogisticRegression& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		LinearModel::operator=(source);
	}
	return *this;
}

// functions
const void LogisticRegression::fit(const mat& x, const mat& y) {
	/*
	Run Newton's Method to minimize J(theta) for logistic regression.
	Args:
	x: Training example inputs.Shape(m, n).
	y : Training example labels.Shape(m, ).
	*/
	arma::uword m = x.n_rows, n = x.n_cols;
	theta = arma::zeros(n); // initate theta with zeros, shape of (m, 1)
	
	// apply newton's method
	int n_iter = 0;
	
	while (n_iter < max_iter) {
		mat h_x = util::sigmoid(x * theta);

		mat gradient_J = -x.t() * (y - h_x) / m;
		mat H_J = (x.each_col() % (h_x % (1 - h_x))).t() * x / m;
		// update theta & check if coverge
		vec step = arma::inv(H_J) * gradient_J;
		theta -= step;
		if (arma::norm(step, 1) < eps) { break; };
		n_iter += 1;
	}
	return;
}

const mat LogisticRegression::predict(const mat& x, const bool& p) {
	/*
	Make a prediction given new inputs x.
	Args:
	x: Inputs of shape(m, n).
	p : if return prob of the output

	Returns :
	Outputs of shape(m, 1);
	*/

	// hypothestic function results with trained theta
	mat h_x_opt = util::sigmoid(x * theta);
	arma::uword m = h_x_opt.n_rows;

	if (p) { return h_x_opt; }
	else {
		mat ks(m, 1);
		for (int i = 0; i < m; i++) {
			if (h_x_opt[i] >= 0.5) { ks[i] = 1; }
			else { ks[i] = 0; };
		}
		return ks;
	}
}

// executive function of p01b_logreg
void p01b_logreg(string dataset) {
	// loadata set
	std::string root, path_train, path_eval, savedr;
	root = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\data\\";
	savedr = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\src_cpp\\src_cpp\\output\\";
	path_train = dataset + "_train.csv";
	path_eval = dataset + "_valid.csv";

	tuple<mat, mat> data_train = util::load_dataset(root + path_train);
	tuple<mat, mat> data_eval = util::load_dataset(root + path_eval);

	mat X_train, y_train, X_eval, y_eval;

	// Get train/eval data
	X_train = get<0>(data_train);
	y_train = get<1>(data_train);

	X_eval = get<0>(data_eval);
	y_eval = get<1>(data_eval);

	// train dataset1, predict, save
	LogisticRegression logreg;
	logreg.fit(X_train, y_train);

	mat y_pred;
	y_pred = logreg.predict(X_eval);
	y_pred.save(savedr + "p01b_pred_logreg_" + dataset + ".csv", arma::arma_ascii);

	return;
}