// p05b_lwr source file to apply class and function

#include "p05b_lwr.hpp"
#include "util.hpp"


LocallyWeightedLinearRegression::LocallyWeightedLinearRegression() : LinearModel(), tau(0.5) {};
LocallyWeightedLinearRegression::LocallyWeightedLinearRegression(const LocallyWeightedLinearRegression& source) : LinearModel(source), tau(source.tau) {};
LocallyWeightedLinearRegression::LocallyWeightedLinearRegression(const double& tau, mat& theta, const double& step_size, const int& max_iter, const double& eps) : LinearModel(theta, step_size, max_iter, eps), tau(tau) {};
LocallyWeightedLinearRegression::~LocallyWeightedLinearRegression() {};

// assignment operator
LocallyWeightedLinearRegression& LocallyWeightedLinearRegression::operator=(const LocallyWeightedLinearRegression& source) {
	if (this == &source) {
		cout << "self-assignment check" << endl;
	}
	else {
		LinearModel::operator=(source);
		tau = source.tau;
	}
	return *this;
}

// functions
const void LocallyWeightedLinearRegression::fit(const mat& x, const mat& y) {
	// train the model with the dataset, just to save the inputs as memember data
	m_x = x;
	m_y = y;
	return;
}

const mat LocallyWeightedLinearRegression::predict(const mat& x, const bool& p) {
	/*
	Make predictions given inputs x.
	Args:
	x: Inputs of shape(m, n).

	Returns :
	Outputs of shape(m, 1).
	*/
	
	arma::uword m = x.n_rows, n = x.n_cols;
	arma::uword xm = m_x.n_rows, xn = m_x.n_cols;
	mat y_pred(m, 1);

	for (int i = 0; i != m; i++) {
		mat W; // weighted matrix to store weighted values
		mat normed(xm, 1); // norm order 2 for each row
		for (int j = 0; j != xm; j++) {
			normed.row(j) = norm(m_x.row(j) - x.row(i), 2);
		}

		W = diagmat(exp(-normed / (2 * tau * tau)));

		theta = arma::inv(m_x.t() * W * m_x) * m_x.t() * W * m_y;
		y_pred.row(i) = x.row(i) * theta;
	}
	return y_pred;
}

void p05b_lwr (string dataset) {
	// loadata set
	std::string root, path_train, path_eval, savedr;
	root = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\data\\";
	savedr = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\src_cpp\\src_cpp\\output\\";
	path_train = dataset + "_train.csv";
	path_eval = dataset + "_valid.csv";

	tuple<mat, mat> data_train = util::load_dataset(root + path_train, 1, 1, true);
	tuple<mat, mat> data_eval = util::load_dataset(root + path_eval, 1, 1, true);

	// Get train/eval data
	mat X_train, X_eval;
	vec y_train, y_eval;

	X_train = get<0>(data_train);
	y_train = get<1>(data_train);

	X_eval = get<0>(data_eval);
	y_eval = get<1>(data_eval);

	// initate and train model
	
	LocallyWeightedLinearRegression model_wlr;
	model_wlr.fit(X_train, y_train);
	
	vec y_pred;
	y_pred = model_wlr.predict(X_eval);
	y_pred.save(savedr + "p05b_pred_wlr_" + dataset + ".csv", arma::arma_ascii);

	cout << "MSE with tau=0.5 is " << util::MSE(y_pred, y_eval);
}

void p05c_tau(string dataset, vec taus) {
	// loadata set
	std::string root, path_train, path_eval, path_test, savedr;
	root = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\data\\";
	savedr = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\src_cpp\\src_cpp\\output\\";
	path_train = dataset + "_train.csv";
	path_eval = dataset + "_valid.csv";
	path_test = dataset + "_test.csv";

	tuple<mat, mat> data_train = util::load_dataset(root + path_train, 1, 1, true);
	tuple<mat, mat> data_eval = util::load_dataset(root + path_eval, 1, 1, true);
	tuple<mat, mat> data_test = util::load_dataset(root + path_test, 1, 1, true);

	// Get train/eval data
	mat X_train, X_eval, X_test;
	vec y_train, y_eval, y_test;

	X_train = get<0>(data_train);
	y_train = get<1>(data_train);

	X_eval = get<0>(data_eval);
	y_eval = get<1>(data_eval);

	X_test = get<0>(data_test);
	y_test = get<1>(data_test);

	// initate and train model
	vec y_pred;
	
	arma::uword n_taus = taus.n_rows;
	vec MSEs(n_taus);
	double mse;
	LocallyWeightedLinearRegression model_wlr;
	for (int i = 0; i != n_taus; i ++) {
		model_wlr.tau = as_scalar(taus.row(i));
		model_wlr.fit(X_train, y_train);
		y_pred = model_wlr.predict(X_eval);
		mse = util::MSE(y_pred, y_eval);
		MSEs.row(i) = mse;
	}
	
	arma::uword dx = MSEs.index_min();

	cout << "best tau is " << taus.row(dx) << endl;

	model_wlr.tau = as_scalar(taus.row(dx));
	model_wlr.fit(X_train, y_train);
	y_pred = model_wlr.predict(X_test);
	cout << "MSE on the test dataset is " << util::MSE(y_pred, y_test) << endl;
	y_pred.save(savedr + "p05c_pred_tau_" + dataset + ".csv", arma::arma_ascii);
}