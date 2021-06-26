// run file to go over algos
#include <iostream>
#include <armadillo>
#include "util.hpp"
#include "linear_model.hpp"
#include "p01b_logreg.hpp"
#include "p01e_GDA.hpp"
#include "p03d_poisson.hpp"
#include "p05b_lwr.hpp"


int main() {

	std::string excode= "p05b";

	if (excode == "p01b") {
		p01b_logreg("ds1");
		p01b_logreg("ds2");
	}

	if (excode == "p01e") {
		p01e_GDA("ds1");
		p01e_GDA("ds2");
	}

	if (excode == "p03d") {
		p03d_poisson("ds4");
	}

	if (excode == "p05b") {
		p05b_lwr("ds5");
	}

	if (excode == "all") {
		p01b_logreg("ds1");
		p01b_logreg("ds2");

		p01e_GDA("ds1");
		p01e_GDA("ds2");

		p03d_poisson("ds4");

		p05b_lwr("ds5");
	}
	
	return 0;
}
