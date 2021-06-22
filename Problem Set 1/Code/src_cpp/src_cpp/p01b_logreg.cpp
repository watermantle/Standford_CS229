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