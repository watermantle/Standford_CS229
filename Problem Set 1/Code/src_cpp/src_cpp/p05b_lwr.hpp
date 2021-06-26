/*
header file to apply locally weighted regression (lwr)
*/

#include "linear_model.hpp"
#include "armadillo"
#include <iostream>

using namespace std;
using namespace arma;

class LocallyWeightedLinearRegression : public LinearModel {
private:
	// member data to save training data 
	mat m_x, m_y;
public:
	double tau; // Bandwidth parameter for LWR.

	// constructors
	LocallyWeightedLinearRegression();
	LocallyWeightedLinearRegression(const LocallyWeightedLinearRegression& source);
	LocallyWeightedLinearRegression(const double& tau, mat& theta, const double& step_size, const int& max_iter, const double& eps);
	~LocallyWeightedLinearRegression();

	// assignment operator
	LocallyWeightedLinearRegression& operator=(const LocallyWeightedLinearRegression& source);
	
	// functions
	const void fit(const mat& x, const mat& y);
	const mat predict(const mat& x, const bool& p = false);
};

// main function for problem 05b
void p05b_lwr(string dataest);