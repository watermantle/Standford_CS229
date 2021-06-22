// header file for p01b_logreg, defined fitting method and predict function
#ifndef p0b_logreg_HPP
#define p0b_logreg_HPP

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
	LogisticRegression& operator = (const LogisticRegression& source);

	// functions
	const void fit(const mat& x, const mat& y);
	const mat predict(const mat& x, const bool& p=false);
};

#endif;