#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::Matrix2d mat;
    mat << 1, 2,
           2, 3;

    Eigen::EigenSolver<Eigen::Matrix2d> solver(mat);
    
    if (solver.info() != 0) {
        std::cerr << "Error in eigenvalue computation." << std::endl;
        return -1;
    }

    std::cout << "Eigenvalues:\n" << solver.eigenvalues() << std::endl;
    std::cout << "Eigenvectors:\n" << solver.eigenvectors() << std::endl;

    return 0;
}