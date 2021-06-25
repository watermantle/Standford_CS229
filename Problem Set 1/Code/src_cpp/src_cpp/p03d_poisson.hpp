/*
header file to apply poisson regression with GLM (General Linear Model). As poisson distribution is
in the exponential family
*/

#ifndef p03d_poisson_HPP
#define p03d_poisson_HPP

#include <iostream>
#include <armadillo>
#include "linear_model.hpp"

using namespace std;
using namespace arma;

class PoissonRegression : public LinearModel {
private:
	// no private member data
public:
	// constructors
	PoissonRegression();
	PoissonRegression(const PoissonRegression& source);
	PoissonRegression(mat& theta, const double& step_size, const int& max_iter, const double& eps);
	~PoissonRegression();

	// Assignment operator
	PoissonRegression& operator = (const PoissonRegression& source);

	// functions
	const void fit(const mat& x, const mat& y);
	const mat predict(const mat& x, const bool& p = false);
};

// executive function
void p03d_poisson(string dataset);

#endif
