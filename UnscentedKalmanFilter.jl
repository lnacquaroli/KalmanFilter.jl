
struct UnscentedKF
	xk::Vector
	xp::Vector
	P::AbstractMatrix
	K::AbstractMatrix
	w_mean::Vector
	w_cov::Vector
	X::AbstractMatrix
	w_mean_::Vector
	w_cov_::Vector
	X_::AbstractMatrix
	Pyy::AbstractMatrix
	Pxy::AbstractMatrix
	filter::SquareRootFilter
end

"""
	sol = unscented_kalman(
        y, x₀, f, g, P, R, Q;
        α=1e-3, κ=1.0, β=2.0, sqrt_filter=CholeskySqrt(),
        )

Performs the non-linear adaptive unscented Kalman filter algorithm for the estimation of
a state y. This is the additive (zero mean) noise case.

y:  Vector with one measurement (observation) of m = length(x) variables
x₀: A priori estimation of the variables
f:	State Model
g:	Measurement Model
P:  Initial predicted process covariance matrix
R:  Measurement (observable) covariance matrix
Q:  Process noise covariance matrix

α:  Distance between the sample point and mean point: 1 ≤ α ≤ 1e-3 (default)
κ:  Scaling factor (default 1.0)
β:  Prior knowledge of the distribution (default 2.0, Gaussian process)

sqrt_filter: Square root filtering technique for the covariance matrix.
			 Accepted values are: CholeskyModSqrt() (default), SVDSqrt(), CholeskySqrt()

sol: UnscentedKF type
	xk:  	 Kalman filter state estimation
	xp:  	 Predicted state estimate
    P:   	 Process covariance matrix a posteriori
    K:   	 Kalman gain
	w_mean:  Weights of the mean
	w_cov: 	 Weights of the covariance
	S:		 Sigma matrix, size = (num_variables, 2*num_variables+1)
	w_mean_: Weights of the mean expanded (updated process)
	w_cov_:  Weights of the covariance expanded (updated process)
	S_:      Sigma matrix expanded (updated process), size = (#variables, 2*(2*num_variables+1))
	Pyy:     Covariance matrix measurement upate
	Pxy:	 Cross-covariance matrix measurement upate
	filter:  Filter used for the covariance matrix

"""
function unscented_kalman(
	x::Vector,
    x₀::Vector,
    f::Function,
    g::Function,
    P::AbstractMatrix,
	R::AbstractMatrix,
	Q::AbstractMatrix;
	α::Real=1e-3,
	κ::Real=3-length(x),
	β::Real=2.0,
	sqrt_filter::SquareRootFilter=CholeskyModSqrt(),
	)

	m = length(x₀)

	## Predict process
	# Sigma points and weights
	_X = _sigma_points(x₀, P, sqrt_filter, α, κ, β, m)
	w_m, w_c = _mean_covariance_weights(m; α=α, β=β, κ=κ)
	# Propagate sigma points through the non-linear function
	X_ = _propagate_sigma_nlf(f, _X)
	# Predicted state estimation, a priori,
	x̂ = _estimation(X_, w_m)
	# Predicted process covariance matrix, a priori
	P_ = _covariance_matrix_ukf(P, X_, x̂, w_c, Q)

	## Update process
	# Sigma points a posteriori
	_Y = _sigma_points(X_, R, sqrt_filter, α, κ, β, size(X_, 1))
	w_m_, w_c_ = _mean_covariance_weights(2*m; α=α, β=β, κ=κ)
	# Measurement estimation
	Y_ = _propagate_sigma_nlf(g, _Y)
	ŷ = _estimation(Y_, w_m_)
	# Covariance matrix
	Pyy = _covariance_matrix_ukf(P, Y_, ŷ, w_c_, R)
	# Cross-covariance matrix
	Pxy = _covariance_matrix_ukf(P, _Y, x̂, Y_, ŷ, w_c_)
	# Kalman gain
	K = _kalman_gain(P, Pxy, Pyy)
	# State estimation measurement update (a posteriori)
	xk = muladd(K, (x .- ŷ), x̂)
	# Covariance matrix measurement update (a posteriori)
	P_ .= P_ .- K*Pyy*K'

	return UnscentedKF(
		vec(xk), vec(x̂),
		P_, K,
		vec(w_m), vec(w_c), X_,
		vec(w_m_), vec(w_c_), Y_,
		Pyy, Pxy,
		sqrt_filter,
		)
end

function _sigma_points(
	x::Vector, P::AbstractMatrix, filter, α::Real, β::Real, κ::Real, m::Int64,
	)
	λ = α^2 * (m + κ) - m
	Pi = _square_root_filter(filter, sqrt(m + λ) .* P)
	return hcat(x, x .+ Pi, x .- Pi)
end

function _sigma_points(
	X::Matrix, P::AbstractMatrix, filter, α::Real, β::Real, κ::Real, m::Int64,
	)
	λ = α^2 * (m + κ) - m
	Pi = _square_root_filter(filter, sqrt(m + λ).*P)
	return hcat(X, X[:,1] .+ Pi, X[:,1] .- Pi)
end

function _mean_covariance_weights(m::Int64; α::Real=1e-3, β::Real=2.0, κ::Real=1.0)
	λ = α^2 * (m + κ) - m
	w_m = ones(2*m + 1) .* 0.5 / (m + λ) # mean weights
	w_c = copy(w_m) # covariance weights
	w_m[1] = λ / (m + λ)
	w_c[1] += one(β) + β - α^2
	return w_m, w_c
end

_propagate_sigma_nlf(f::Function, χ::Array) = (x -> f(x))(χ)

_estimation_product(X::Matrix, w::Vector) = ((A, x) -> x' .* A)(X, w)
function _estimation(X::Matrix, w::Vector)
	return vec(sum(_estimation_product(X::Matrix, w::Vector), dims=2))
end

function _covariance_matrix_ukf(P::AbstractMatrix, X::Matrix, x::Vector, w::Vector, N)
	A = X .- x
	return _return_type(P, w' .* A*A' .+ N)
end

function _covariance_matrix_ukf(
	P::AbstractMatrix, X::Matrix, x::Vector, Y::Matrix, y::Vector, w::Vector,
	)
	return _return_type(P, (w' .* (X .- x)) * (Y .- y)')
end

function _kalman_gain(P::AbstractMatrix, Pxy::AbstractMatrix, Pyy::AbstractMatrix)
	return _return_type(P, Pyy \ Pxy)
end
