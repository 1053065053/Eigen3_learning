#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <fftw3.h>
#include <omp.h>

class MatrixReplacement;
using Eigen::SparseMatrix;

namespace Eigen
{
    namespace internal
    {
        // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
        template <>
        struct traits<MatrixReplacement> : public Eigen::internal::traits<Eigen::SparseMatrix<double>>
        {
        };
    }
}

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement>
{
public:
    // Required typedefs, constants, and method:
    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum
    {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return 10; }
    Index cols() const { return 10; }

    template <typename Rhs>
    Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs> &x) const
    {
        return Eigen::Product<MatrixReplacement, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // Custom API:
    MatrixReplacement() : mp_mat(0) {}

    void attachMyMatrix(const SparseMatrix<double> &mat)
    {
        mp_mat = &mat;
    }
    const SparseMatrix<double> my_matrix() const { return *mp_mat; }

private:
    const SparseMatrix<double> *mp_mat;
};

// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen
{
    namespace internal
    {

        template <typename Rhs>
        struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
            : generic_product_impl_base<MatrixReplacement, Rhs, generic_product_impl<MatrixReplacement, Rhs>>
        {
            typedef typename Product<MatrixReplacement, Rhs>::Scalar Scalar;

            template <typename Dest>
            static void scaleAndAddTo(Dest &dst, const MatrixReplacement &lhs, const Rhs &rhs, const Scalar &alpha)
            {
                // This method should implement "dst += alpha * lhs * rhs" inplace,
                // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
                assert(alpha == Scalar(1) && "scaling is not implemented");
                EIGEN_ONLY_USED_FOR_DEBUG(alpha);

                // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
                // but let's do something fancier (and less efficient):

                dst(0) = 2 * rhs(0) - rhs(1);
                for (Index i = 1; i < lhs.cols() - 1; ++i)
                    dst(i) = 2 * rhs(i) - rhs(i - 1) - rhs(i + 1);

                dst(lhs.cols() - 1) = 2 * rhs(lhs.cols() - 1) - rhs(lhs.cols() - 2);
            }
        };

    }
}

class myPreconditioner
{
public:
    typedef Eigen::VectorXd VectorXd;
    VectorXd solve(const VectorXd &input) const
    {
        int n = input.size();
        VectorXd res = input;
        res(0) = input(0) * 10;
        res(n - 1) = input(n - 1) * 10;
        return res;
    }
    // 返回状态信息
    Eigen::ComputationInfo info() { return Eigen::Success; }

    template <typename Rhs>
    void compute(const Rhs &mat)
    {
        // 这里可以实现你的预处理逻辑
    }
};

void gmres_test()
{
    int n = 10;
    Eigen::SparseMatrix<double> S = Eigen::MatrixXd::Random(n, n).sparseView(0.5, 1);
    S = S.transpose() * S;

    MatrixReplacement A;
    A.attachMyMatrix(S);

    Eigen::VectorXd b(n), x;
    b = Eigen::VectorXd::Constant(n, 1);

    // Solve Ax = b using various iterative solver with matrix-free version:
    {
        // Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, myPreconditioner> cg;
        cg.compute(A);
        x = cg.solve(b);
        std::cout << "CG:       #iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;
        std::cout << x.transpose() << std::endl;
    }

    {
        Eigen::GMRES<MatrixReplacement, Eigen::IdentityPreconditioner> gmres;
        gmres.compute(A);
        x = gmres.solve(b);
        std::cout << "GMRES:    #iterations: " << gmres.iterations() << ", estimated error: " << gmres.error() << std::endl;
        std::cout << x.transpose() << std::endl;
    }
}

// do fft for the each row of the n by m real matrix
// note:我发现如果plan已经设定了，把in和out的指针指向其他地方是不行的
void fft_r2c_mr(std::complex<double> *in_p, std::complex<double> *out_p, int n)
{
    fftw_complex *in = reinterpret_cast<fftw_complex *>(in_p);
    fftw_complex *out = reinterpret_cast<fftw_complex *>(out_p);

    fftw_plan plan = fftw_plan_dft_1d(n, in, out,FFTW_FORWARD, FFTW_ESTIMATE);

    
    fftw_execute_dft(plan, in, out);


    fftw_destroy_plan(plan);
    // 在函数调用结束前清理
    fftw_cleanup();

    out_p = reinterpret_cast<std::complex<double> *>(out_p);

}

void ifft_c2c_mr(std::complex<double> *in_p, std::complex<double> *out_p, int n)
{
    // double *in = matrix_in.data();
    fftw_complex *out = reinterpret_cast<fftw_complex *>(out_p);
    fftw_complex *in = reinterpret_cast<fftw_complex *>(in_p);

    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

    
    fftw_execute_dft(plan, in, out);


    fftw_destroy_plan(plan);
    // 在函数调用结束前清理
    fftw_cleanup();

    out_p = reinterpret_cast<std::complex<double> *>(out_p);

}

void Eigen_vector_test()
{
    using Eigen::VectorXd;

    int n = 10;
    VectorXd vec1;
    vec1.setLinSpaced(n, 0, 9);

    double *vec2 = vec1.data();

    for (int i = 0; i < n; i++)
    {
        std::cout << vec2[i] << " ";
        vec2[i] = i + 1;
    }
    std::cout << std::endl;
    std::cout << vec1.transpose() << std::endl;

    vec2 = nullptr; // 好习惯
}

void fft_test()
{
    using Eigen::VectorXcd;

    int n = 10;
    VectorXcd vec1;
    Eigen::VectorXcd vec_res(n), vec_res2(n);
    vec1.setLinSpaced(n, 0, 9);

    fft_r2c_mr(vec1.data(), vec_res.data(), n);

    std::cout << vec1.transpose() << std::endl;
    std::cout << vec_res.transpose() << std::endl;

    ifft_c2c_mr(vec_res.data(), vec_res2.data(), n);

     std::cout << vec_res2.transpose()/n << std::endl;
}

int main()
{
    // gmres_test();
    //  Eigen_vector_test();
    fft_test();
    return 0;
}