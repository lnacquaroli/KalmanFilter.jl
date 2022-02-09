
struct LinearKF
    xk::Vector
	xp::Vector
	P::AbstractMatrix
	K::AbstractMatrix
	S::AbstractMatrix
	y::Vector
	filter::SquareRootFilter
end

"""
	sol = linear_kalman(
		x, x₀, P, F, H, Q, R;
		B=zeros(length(x), length(x)),
		u=zeros(length(x)),
		w=zeros(length(x)),
		v=zeros(length(x)),
		sqrt_filter=CholeskySqrt(),
		)

Performs the simple Kalman filter algorithm for the estimation of a state x.

x:   Vector with one measurement (observation) of m = length(x) variables
x₀:  A priori estimation of the variables
P:   Initial predicted process covariance matrix
F:	 Transition matrix (linear state model)
H:	 Measurement matrix (linear measurement model)
Q:   Process noise covariance matrix
R:   Measurement (observable) covariance matrix

B:	 Control-input matrix (linear state model)
u:	 Control vector (linear state model)
w:	 Process noise vector, w ~ N(0,Q)
v:	 Measurement (observable) noise vector, v ~ N(0,R)

sqrt_filter: Square root filtering technique for the covariance matrix.
			 Accepted values are: CholeskyModSqrt() (default), SVDSqrt(), CholeskySqrt()

sol: LinearKalman type
    xk:     Kalman filter state estimation
    xp:     Predicted state estimate
    P:      Process covariance matrix a posteriori
    K:      Kalman gain
	S:      Innovation (or pre-fit residual) covariance
	y:      Innovation or measurement pre-fit residual
	filter: Type of SquareRootFilter selected

"""
function linear_kalman(
	x::Vector,
    x₀::Vector,
    P::AbstractMatrix,
	F::AbstractMatrix,
	H::AbstractMatrix,
	Q::AbstractMatrix,
	R::AbstractMatrix;
	B::AbstractMatrix=zeros(length(x), length(x)),
	u::Union{Vector, Real}=zeros(length(x)),
	w::Vector=zeros(length(x)),
	v::Vector=zeros(length(x)),
	sqrt_filter::SquareRootFilter=CholeskyModSqrt(),
	)

	m = length(x)

	## Predict process
	# State estimate propagation (Predicted state, a priori)
	x̂ = _linear_state_model(x₀, F, B, u, w)
	# Error covariance propagation (Predicted process covariance matrix, a priori)
	P_ = _process_covariance_priori(P, F, Q)
	# Innovation (or pre-fit residual) covariance
	PHt = P_ * H'
	S = _innovation_covariance(P_, PHt, H, R)
	# Kalman gain
	K = _kalman_gain(P_, PHt, S)

	## Update process
	# New observation (innovation process)
    y = _linear_observation_model(x, H, v)
	# Innovation or measurement pre-fit residual
	y = muladd(-H, x̂, y)
	# State estimate update (Current state, a posteriori)
	xₖ = muladd(K, y, x̂)
	# Error covariance update (Process covariance matrix, a posteriori)
	P_ = _process_covariance_posteriori(P_, H, K, m)

	return LinearKF(vec(xₖ), vec(x̂), P_, K, S, vec(y), sqrt_filter)
end

function _linear_state_model(
	x::Vector, F::AbstractMatrix, B::AbstractMatrix, u::Union{Vector, Real}, w::Vector,
	)
	return vec(F*x .+ B .* u .+ w)
end

function _linear_observation_model(x::Vector, H::AbstractMatrix, v::Vector)
	return vec(muladd(H, x, v))
end

function _process_covariance_priori(P::AbstractMatrix, F::AbstractMatrix, Q::AbstractMatrix)
	return _return_type(P, F*P*F' .+ Q)
end

function _process_covariance_posteriori(
	P::AbstractMatrix, H::AbstractMatrix, K::AbstractMatrix, m::Int64,
	)
	return _return_type(P, (eye(m) .- K*H) *P)
end

function _kalman_gain(P::AbstractMatrix, PHt::AbstractMatrix, S::AbstractMatrix)
	F = qr(S)
	return _return_type(P, F.R \ (F.Q' * PHt))
end

function _innovation_covariance(
	P::AbstractMatrix, PHt::AbstractMatrix, H::AbstractMatrix, R::AbstractMatrix,
	)
	return _return_type(P, H*PHt .+ R)
end
