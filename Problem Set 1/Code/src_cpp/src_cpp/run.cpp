#include "util.hpp"
#include <iostream>
#include <armadillo>



int main() {
	std::string path;
	path = "C:\\Users\\YB\\Documents\\GitHub\\Standford_CS229\\Problem Set 1\\Code\\data\\ds1_train.csv";

	tuple<mat, mat> data = load_dataset(path);

	mat inputs = get<0>(data);
	mat label = get<1>(data);
	
	cout << inputs << endl;

	return 0;
}