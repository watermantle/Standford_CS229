// header file for p01b_logreg, defined fitting method and predict function
#pragma once

#include "linear_model.hpp"

using namespace std;

class LogisticRegression : public LinearModel {
private:
	// no member data
public:
	// constructors
	LogisticRegression();
	LogisticRegression(const LogisticRegression& source);
	LogisticRegression(mat& theta, const double& step_size, const int& max_iter, const double& eps);
	~LogisticRegression();

	// assignment operator
	LogisticRegression& operator = (LogisticRegression&& source);

	// functions
	const void fit(const mat& x, const mat& y);
	const mat predict(const mat& x, const bool& p=false);
};

// p01b_logreg executive function
void p01b_logreg(string dataset);
