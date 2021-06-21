/*
source file to apply class in LinearModel header file
*/
#include <iostream>
#include <armadillo>
#include "linear_model.hpp"

using namespace std;
using namespace arma;

// constructors
LinearModel::LinearModel() : theta(arma::zeros(3)), step_size(0.2), max_iter(100), eps(1e-5) {};
LinearModel::LinearModel(const LinearModel& source) : theta(source.theta), step_size(source.step_size), max_iter(source.max_iter), eps(source.eps) {};
LinearModel::LinearModel(const mat& theta, const double& step_szie, const int& max_iter, const double& eps) : theta(theta), step_size(step_szie), max_iter(max_iter), eps(eps) {}
const void LinearModel::fit(const mat& x, const mat& y)
{
	return void();
}
const mat LinearModel::predict(const mat& x)
{
	return mat();
}
;
LinearModel::~LinearModel() {};

// assignment operation
LinearModel& LinearModel::operator=(const LinearModel& source) {
	if (this == &source) {
		cout << "self=assignment checked";
	}
	else {
		theta = source.theta;
		step_size = source.step_size;
		max_iter = source.max_iter;
		eps = source.eps;
	}
	return *this;
}

// pure virtual functions for fit and predict function, no definations are needed.
