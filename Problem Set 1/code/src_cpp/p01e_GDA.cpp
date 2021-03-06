/*
GDA source file to apply GDA method and functions
*/

#include <armadillo>
#include <filesystem>
#include "p01e_GDA.hpp"
#include "util.hpp"

namespace fs = filesystem;
using filesystem::current_path;
using namespace std;
using namespace arma;

// constructors
GDA::GDA() : LinearModel() {};
GDA::GDA(const GDA& source) : LinearModel(source) {};
GDA::GDA(mat& theta, const double& step_size, const int& max_iter, const double& eps) : LinearModel(theta, step_size, max_iter, eps) {};
GDA::~GDA() {};

// assignment operator
GDA& GDA::operator=(GDA&& source) {
	if (this == &source) {
		cout << "self-assignment checked";
	}
	else {
		LinearModel::operator=(source);
	}
	return *this;
}

// functions
// fit function to train the theta
const void GDA::fit(const mat& x, const mat& y) {
	/*
	Fit a GDA model to training set given by x and y
	Args:
	x: Training example inputs.Shape(m, n).
	y : Training example labels.Shape(m, ).

	Returns :
	theta : GDA model parameters.
	*/
	unsigned int m = as_scalar(x.n_rows), n = as_scalar(x.n_cols);

	// calculate GDA parameters
	double num_y1 = accu(y);
	double phi = num_y1 / m;

	// x|y=0 and x|y=1
	mat x_y0 = x.rows(find(y == 0));
	mat x_y1 = x.rows(find(y == 1));

	mat mu0 = sum(x_y0, 0) / (m - num_y1);
	mat mu1 = sum(x_y1, 0) / num_y1;
	
	// sigma calculation
	mat sigma0 = (x_y0.each_row() - mu0).t() * (x_y0.each_row() - mu0);
	mat sigma1 = (x_y1.each_row() - mu1).t() * (x_y1.each_row() - mu1);
	mat sigma = (sigma0 + sigma1) / m;

	// theta calculation with GDA parameters
	theta = arma::inv(sigma) * (mu1 - mu0).t();
	mat theta0 = -0.5 * (mu1 - mu0) * inv(sigma) * (mu1 + mu0).t() - log((1 - phi) / phi);
	theta = join_cols(theta0, theta);
}

// prediction function
const mat GDA::predict(const mat& x, const bool& p) {
	/*
	Make a prediction given new inputs x.

	Args:
	x: Inputs of shape(m, n).
	p : if return prob of the outcomes
	Returns :
	Outputs of shape(m, 1)
	*/

	// hypothestic function results with trained theta
	mat x_theta = x * theta;
	mat h_x_opt = util::sigmoid(x_theta);
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
void p01e_GDA(string dataset) {
	// loadata set
	std::string root, path_train, path_eval, savedr;
	fs::path p = current_path();
	root = p.parent_path().parent_path().string() + R"(/data/)";
	savedr = p.string() + R"(/output/)";

	path_train = dataset + "_train.csv";
	path_eval = dataset + "_valid.csv";

	tuple<mat, mat> data_train = util::load_dataset(root + path_train, 2, 2, false);
	tuple<mat, mat> data_eval = util::load_dataset(root + path_eval);

	mat X_train, y_train, X_eval, y_eval;

	// Get train/eval data
	X_train = get<0>(data_train);
	y_train = get<1>(data_train);

	X_eval = get<0>(data_eval);
	y_eval = get<1>(data_eval);

	// train dataset1, predict, save
	
	auto GDA_model_ptr = make_unique<GDA>();
	(*GDA_model_ptr).fit(X_train, y_train);

	mat y_pred;
	y_pred = (*GDA_model_ptr).predict(X_eval);
	y_pred.save(savedr + "p01e_pred_GDA_" + dataset + ".csv", arma::arma_ascii);

	return;
}