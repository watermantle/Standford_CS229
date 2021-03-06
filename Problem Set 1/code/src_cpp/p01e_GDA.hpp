/*
header file to apply GDA (Gaussian Discriminant Analysis)
*/
#pragma once

#include <iostream>
#include <armadillo>
#include "util.hpp"
#include "linear_model.hpp"


class GDA : public LinearModel {
private:
	// no private member data
public:
	GDA();
	GDA(const GDA& source);
	GDA(mat& theta, const double& step_size, const int& max_iter, const double& eps);
	~GDA();

	// assignment operator
	GDA& operator = (GDA&& source);

	// functions
	const void fit(const mat& x, const mat& y);
	const mat predict(const mat& x, const bool& p = false);
};

void p01e_GDA(string dataset);