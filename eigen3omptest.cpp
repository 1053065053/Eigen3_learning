#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <chrono>

void eigen_test(int num_threads)
{
    using Eigen::MatrixXd;
    using Eigen::seqN;
    using Eigen::VectorXd;

    int block_size = 5000;
    int num_block = 1000;

    MatrixXd A = MatrixXd::Random(block_size, block_size);
    VectorXd b = VectorXd::Random(block_size * num_block);
    VectorXd res = VectorXd::Constant(block_size * num_block, 0);

    // std::cout << A << std::endl;
    // std::cout << b << std::endl;
    // std::cout << res << std::endl;

auto start = std::chrono::high_resolution_clock::now();
    omp_set_num_threads(num_threads);
#pragma omp parallel for
    for (int i = 0; i < num_block; i++)
    {
        // res(seqN(i * block_size, block_size)) = A * b(seqN(i * block_size, block_size));
        res.segment(i * block_size, block_size) =A * b.segment(i * block_size, block_size);
    }
    // std::cout << res << std::endl;
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "执行时间: " << duration.count() << " 秒" << std::endl;

}

int main(int argc, char **argv)
{
    // int num_threads = 1;
    // if(argc>1){
    //     num_threads = std::atoi(argv[1]);
    // }
    // eigen_test(num_threads);

    Eigen::Matrix<int,5,1> m(1,2,4,8,16);
    for(auto num_threads: m){
        eigen_test(num_threads);
    }

    return 0;
}