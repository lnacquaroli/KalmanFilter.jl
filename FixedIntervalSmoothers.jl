
abstract type Smoothers end
struct RTTSmoother <: Smoothers end
RauchTungStriebel() = RTTSmoother()
struct MBFmoother <: Smoothers end
ModifiedBrysonFrazier() = MBFmoother()


## Rauch Tung Striebel
struct RTSSmoother
	xk::Vector
	P::AbstractMatrix
end

function kalman_smooth(
	s::RTTSmoother,
	x₀::Vector,
	xk::Vector,
	xkp1::Vector,
	Pk::AbstractMatrix,
	Pkp1::AbstractMatrix,
	F::AbstractMatrix,
	)
	A = similar(F)
	mul!(A, Pk, F')
	A = _return_type(Pk, Pkp1 \ A)
	P = Pkp1 .- Pk
    P = Pk .+ _return_type(Pk, A*P*A')
	x̂ = x₀ .- xkp1
	x̂ .= muladd(A, x̂, xk)
    return RTSSmoother(vec(x̂), P)
end

## Modified Bryson–Frazier smoother
struct MBFSmoother
	xk::Vector
	P::AbstractMatrix
	L::AbstractMatrix
	l::Vector
end

function kalman_smooth(
	s::MBFmoother,
	xk::Vector,
	Pk::AbstractMatrix,
	K::AbstractMatrix,
	F::AbstractMatrix,
	H::AbstractMatrix,
	Λ::AbstractMatrix,
	λ::Vector,
	S::AbstractMatrix,
	y::Vector,
	)
	A = _return_type(Pk, muladd(-K, H, eye(length(xk))))
	Aᵀ, Fᵀ = A', F'
	SH = S \ H'
	λ̂ = mul!(similar(λ), Fᵀ, λ)
	Λ̂ = Fᵀ*Λ*F
	x̂ = muladd(Pk, -λ̂, xk)
	P = _return_type(Pk, Pk .- Pk*Λ̂*Pk)
	Λ̂ = muladd(SH, H, Aᵀ*Λ̂*A)
	λ̂ = muladd(Aᵀ, λ̂, -SH*y)
	return MBFSmoother(vec(x̂), P, Λ̂, vec(λ̂))
end
