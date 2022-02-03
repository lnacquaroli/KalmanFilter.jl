
abstract type SquareRootFilter end

struct CholeskyFactorization <: SquareRootFilter end
CholeskySqrt() = CholeskyFactorization()

struct SVDFactorization <: SquareRootFilter end
SVDSqrt() = SVDFactorization()

struct ModifiedCholeskyFactorization <: SquareRootFilter end
CholeskyModSqrt() = ModifiedCholeskyFactorization()
