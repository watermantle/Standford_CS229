/*
header file for basic linear model
base class providing some basic functionalities
*/

#pragma once

#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;


class LinearModel {
private:
	// no private variables for convenience pupose.

public:
	/*
	Args :
	step_size: Step size for iterative solvers only.
	max_iter : Maximum number of iterations for the solver.
	eps : Threshold for determining convergence.
	theta_0 : Initial guess for theta.If None, use the zero vector.
	*/

	vec theta;
	double step_size, eps;
	int max_iter;

	// constructors
	LinearModel(); // default constructor
	LinearModel(const LinearModel& source); // copy constructor
	LinearModel(mat& theta, const double& step_size, const int& max_iter, const double& eps); // constructor given value
	virtual ~LinearModel();

	// Assignment operator
	LinearModel& operator = (const LinearModel& source);

	// functions
	virtual const void fit(const mat& x, const mat& y) = 0; // pure virtual function, different for different models
	virtual const mat predict(const mat& x, const bool& p = false) = 0; // pure virtual function for prediction, different for different models
};
