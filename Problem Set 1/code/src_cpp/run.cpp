// run file to go over algos
#include <iostream>
#include <armadillo>
#include <chrono>
#include "util.hpp"
#include "linear_model.hpp"
#include "p01b_logreg.hpp"
#include "p01e_GDA.hpp"
#include "p03d_poisson.hpp"
#include "p05b_lwr.hpp"

using namespace std;
using namespace std::chrono;

int main() {
	auto start = high_resolution_clock::now();

	std::string excode= "all";

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

	if (excode == "p05c") {
		p05c_tau("ds5", vec{ 5e-2, 1e-1, 5e-1, 1e0, 1e1 });
	}

	if (excode == "all") {
		p01b_logreg("ds1");
		p01b_logreg("ds2");

		p01e_GDA("ds1");
		p01e_GDA("ds2");

		p03d_poisson("ds4");

		p05b_lwr("ds5");
	}

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop - start);

	cout << "the program running time is " << duration.count() << endl;

	return 0;
}
