
# Identity matrix
eye(m::Int64) = Diagonal(ones(m))

unpack(s, f::Symbol) = reduce(hcat, getproperty.(s, f))

function unpack(s, f::Vector{Symbol})
    A = []
    for i in f
        push!(A, reduce(hcat, getproperty.(s, i)))
    end
    return A
end

unpack_matrix(s, f) = getproperty.(s, f)

function unpack_matrix(s, f::Vector{Symbol})
    A = []
    for i in f
        push!(A, getproperty.(s, i))
    end
    return A
end

_return_type(P::Matrix, C) = Matrix(C)
_return_type(P, C) = Diagonal(C)

_square_root_filter(filter::CholeskyFactorization, P) = _return_type(P, cholesky(P).L)

function _square_root_filter(filter::ModifiedCholeskyFactorization, P)
	F = _modified_cholesky(P)
	_return_type(P, F.U * sqrt.(F.D))
end

function _square_root_filter(filter::SVDFactorization, P)
	_C = svd(Matrix(P))
	return _return_type(P, _C.U * sqrt(Diagonal(_C.S)))
end

"""

	F = _modified_cholesky(P)

Computes modified Cholesky factors F.U and F.D of a symmetric positive definite matrix P,
such that F.U is unit upper triangular and F.D is diagonal. Thus, and P = F.U*F.D*F.U'.

Reference: Grewal - Kalman filtering theory and practice

"""
function _modified_cholesky(P::AbstractMatrix)
	isposdef(P) || throw("input matrix must be positive-definite. Modified Cholesky factorization failed.")
	n = size(P, 1)
	_P = (P .+ P') ./ 2.0 # take symmetric part
	U, D = similar(_P), similar(_P)
	for j in n:-1:1
   		for i in j:-1:1
			s = P[i, j]
			for k in j+1:n
				s -= U[i, k]*D[k, k]*U[j, k]
			end
			if i==j
				D[j, j] = s
				U[j, j] = 1.0
			else
				U[i, j] = s / D[j, j]
			end
		end
	end
	return (U=U, D=D)
end
