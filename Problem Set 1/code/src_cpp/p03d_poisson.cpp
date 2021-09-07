/*
source file to apply functions and class in the header file
*/

#include "p03d_poisson.hpp"
#include <filesystem>
#include "util.hpp"

namespace fs = filesystem;
using filesystem::current_path;

// constructors
PoissonRegression::PoissonRegression() : LinearModel() {};
PoissonRegression::PoissonRegression(const PoissonRegression& source) : LinearModel(source) {};
PoissonRegression::PoissonRegression(mat& theta, const double& step_size, const int& max_iter, const double& eps) : LinearModel(theta, step_size, max_iter, eps) {};
PoissonRegression::~PoissonRegression() {};

// Assignment operator
PoissonRegression& PoissonRegression::operator=(PoissonRegression&& source) {
	if (this == &source) {
		cout << "self-assignment check" << endl;
	}
	else { LinearModel::operator=(source);}
	return *this;
}

// functions
const void PoissonRegression::fit(const mat& x, const mat& y) {
	/*Run gradient ascent to maximize likelihood for Poisson regression.
	Args:
	x: Training example inputs.Shape(m, n).
	y : Training example labels.Shape(m, 1).*/

	// initiate theta with zeros
	uword m = x.n_rows, n = x.n_cols, batch_size=100;
	theta = arma::zeros(n, 1);

	while (true) {
		// randomly pick up indices
		uvec idx_rnd = randi<uvec>(batch_size, 1, distr_param(0, m - 1));
		mat xi = x.rows(idx_rnd);
		mat yi = y.rows(idx_rnd);
		mat gradient = xi.t() * (yi - exp(xi * theta)) / batch_size;

		theta += step_size * gradient;
		if (arma::norm(step_size * gradient, 1) < eps) { break; }
	}
}

const mat PoissonRegression::predict(const mat& x, const bool& p) {
	/*
	Make a prediction given inputs x.
	Args:
	x: Inputs of shape(m, n).
	Returns :
	Floating - point prediction for each input, shape(m, 1).*/
	if (p) { cout << "Prob is not supported in this model, only predicted values returned.\n"; }
	return exp(x * theta);
}

void p03d_poisson(string dataset) {
	// loadata set
	std::string root, path_train, path_eval, savedr;
	fs::path p = current_path();
	root = p.parent_path().parent_path().string() + R"(/data/)";
	savedr = p.string() + R"(/output/)";

	path_train = dataset + "_train.csv";
	path_eval = dataset + "_valid.csv";

	tuple<mat, mat> data_train = util::load_dataset(root + path_train, 4, 4, false);
	tuple<mat, mat> data_eval = util::load_dataset(root + path_eval, 4, 4, false);

	// Get train/eval data
	mat X_train, y_train, X_eval, y_eval;

	X_train = get<0>(data_train);
	y_train = get<1>(data_train);

	X_eval = get<0>(data_eval);
	y_eval = get<1>(data_eval);

	// train dataset1, predict, save
	mat theta = arma::zeros(5);
	double step_size = 1e-7;
	unsigned int max_iter = 100;
	double eps = 1e-5;

	PoissonRegression pois_reg(theta, step_size, max_iter, eps);
	auto PoissonReression_ptr = make_unique<PoissonRegression>(pois_reg);
	

	(*PoissonReression_ptr).fit(X_train, y_train);

	mat y_pred;
	y_pred = (*PoissonReression_ptr).predict(X_eval);
	y_pred.save(savedr + "p03d_pred_poisson_" + dataset + ".csv", arma::arma_ascii);
}