// run file to go over algos
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
		p01b_logreg("ds1");
		p01b_logreg("ds2");
	}

	if (excode == "p01e") {
		p01e_GDA("ds1");
		p01e_GDA("ds2");
	}
}
