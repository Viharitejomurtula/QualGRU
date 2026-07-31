#include <iostream>
#include <Eigen/Dense>


int main() {
	Eigen::VectorXf v(3);
	v << 1.0, 2.0, 3.0;
	std::cout << "vector:\n" << v << std::endl;
	return 0;
}
