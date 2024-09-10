#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include <iostream>

class MatrixReplacement;

namespace Eigen
{
    namespace internal
    {
        template <>
        struct traits<MatrixReplacement> : public Eigen::internal::traits<Eigen::SparseMatrix<double>>
        {
        };
    }
}

class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement>
{
    // necessary
public:
    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum
    {
        ColsAtCompileTime = Eigen::Dynamic,
        RowsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Index rows() const { return num_rows; }
    Index cols() const { return num_cols; }

      template<typename Rhs>
  Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }

    template <typename Rhs>
    Eigen::MatrixBase<Rhs> myproduct(const Eigen::MatrixBase<Rhs> &x) const
    {
        Eigen::MatrixBase<Rhs> res;
        res(1) = b * x(1) + c * x(2);
        for (int i = 1; i < this->cols() - 1; i++)
        {
            res(i) = a * x(i - 1) + b * x(i) + c * x(i + 1);
        }
        res(this->cols()) = a * x(this->cols() - 1) + b * x(this->cols());
        return res;
    }

    // customed
    MatrixReplacement() {};
    MatrixReplacement(double aa, double bb, double cc, int dimension) : a(aa), b(bb), c(cc), num_rows(dimension), num_cols(dimension) {}

private:
    double a, b, c;
    int num_rows;
    int num_cols;
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
                dst=lhs.myproduct(rhs);
            }
        };

    }
}

int main()
{
    using Eigen::VectorXd;

    int n = 10;
    VectorXd rhs = VectorXd::Constant(n, 1);

    double a = -1, b = 2, c = -1;
    MatrixReplacement A(a, b, c, n);

    // Solve Ax = b using various iterative solver with matrix-free version:
    {
        Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        cg.compute(A);
        VectorXd x = cg.solve(rhs);
        std::cout << "CG:       #iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;
    }
}
