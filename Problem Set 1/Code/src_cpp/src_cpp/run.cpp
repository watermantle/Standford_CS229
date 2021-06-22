
#include <iostream>
#include <armadillo>
#include "util.hpp"
#include "linear_model.hpp"
#include "p01b_logreg.hpp"
#include "p01e_GDA.hpp"




int main() {
	std::string root, path1_train, path1_eval, path2_train, path2_eval, savedr;
	root = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\data\\";
	savedr = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\src_cpp\\src_cpp\\output\\";
	path1_train = "ds1_train.csv";
	path1_eval = "ds1_valid.csv";
	path2_train = "ds2_train.csv";
	path2_eval = "ds2_valid.csv";

	std::string excode= "p01e";

	if (excode == "p01b") {
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

		y_pred.save(savedr + "p01b_pred_logreg_ds1.csv", arma::arma_ascii);
	}

	if (excode == "p01e") {
		tuple<mat, mat> data_train = util::load_dataset(root + path1_train, 2, 2, false);
		tuple<mat, mat> data_eval = util::load_dataset(root + path1_eval);
		mat X_train, y_train, X_eval, y_eval;

		X_train = get<0>(data_train);
		y_train = get<1>(data_train);

		X_eval = get<0>(data_eval);
		y_eval = get<1>(data_eval);

		GDA GDA_model;
		GDA_model.fit(X_train, y_train);

		mat y_pred;
		y_pred = GDA_model.predict(X_eval);

		y_pred.save(savedr + "p01e_pred_GDA_ds1.csv", arma::arma_ascii);
	}
	return 0;
}
