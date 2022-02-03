
##
import LinearAlgebra as linalg
import Plots as plt

include("KalmanFilter.jl")
import .KalmanFilter as kf

##
function get_data()
    # Observations
    x = [4000.0 4260.0 4550.0 4860.0 5110.0] # position [m]
    v = [ 280.0  282.0  285.0  286.0  290.0] # velocity [m/s]
    # x = [4000.0 4260.0 4550.0 4860.0 5110.0 5220.0 5115.0 4995.0 4871.0] # position [m]
    # v = [ 280.0  282.0  285.0  286.0  290.0  295.0  292.0  287.0  284.0] # velocity [m/s]
    X = [x; v]

    # Initial values
    a = 2.0 # acceleration
    Δt = 1.0 # Time step
    ΔP = [20.0; 5.0] # Process errors in process covariance matrix
    Δ = [25.0; 6.0] # Observation errors
    return X, a, Δ, ΔP, Δt
end

## Define the state space model

function define_model(X, a, Δ, ΔP, Δt)
    m, n = size(X)
    Q = zeros(m, m) # Process noise covariance matrix
    F = [1.0 Δt; 0.0 1.0] # Transition matrix
    B = [Δt^2/2.0 Δt]' # Control-input matrix
    u = 2.0 # # Control vector, acceleration
    R = linalg.Diagonal([Δ[1]^2 0.0; 0.0 Δ[2]^2]) # Sensor noise covariance matrix
    H = kf.eye(m) # Measurement matrix

    # Initial predicted process covariance matrix
    P = linalg.Diagonal([ΔP[1]^2 ΔP[1]*ΔP[2]; ΔP[2]*ΔP[1] ΔP[2]^2])
    return P, Q, R, F, H, B, u, m, n
end

## Data input
function data_input(X, n)
    X = [ [X[1,i], X[2,i]] for i in 1:n ]
    Xp = [ X[1] ] # Initialize predicted state
    Xk = copy(Xp) # Initialized new state matrix
    return Xp, Xk, X
end

## Smoothers
function run_smoothers_bts(Xk, F, Σ)
    k = length(Xk)
    sm = Vector{kf.RTSSmoother}(undef, k)
    sm[k] = kf.RTSSmoother(Xk[k], Σ[k])
    ksmoother = kf.RauchTungStriebel()
    while k > 1
        s = kf.kalman_smooth(ksmoother, sm[k].xk, Xk[k-1], Xk[k], Σ[k-1], Σ[k], F)
        k -= 1
        sm[k] = s
    end
    return sm
end

## UKF
function run_ukf(X, f, g, P, R, Q, α; filter=kf.CholeskySqrt())
    m, n = size(X)
    _, _, Xx = data_input(X, n)
    k = 1
    K = kf._kalman_gain(P, kf.eye(m), R)
    sukf = [ kf.UnscentedKF(
                Xx[k], Xx[k],
                P, K,
                zeros(2*m+1), zeros(2*m+1), zeros(m, 2*m+1),
                zeros((2*m)*2+1), zeros((2*m)*2+1), zeros(m, (2*m)*2+1),
                P, P,
                filter,
            )
    ]
    while k < n
        s = kf.unscented_kalman(
            Xx[k+1], sukf[k].xk, f, g, sukf[k].P, R, Q; sqrt_filter=filter, α=α,
        )
        push!(sukf, s)
        k += 1
    end
    return sukf
end

## Plots
function plot_kf(Xp, Xk, Xx, Δ)
    # Xp, Xk, Xx = reduce(hcat, Xp), reduce(hcat, Xk), reduce(hcat, Xx)
    p1 = plt.plot(Xp[1,:], m=:o, label="Predicted", legend=:topleft, ylabel="Position [m]", xlabel="Time [s]")
    plt.plot!(p1, Xx[1,:], m=:s, label="Measured", ribbon=Δ[1])
    plt.plot!(p1, Xk[1,:], m=:p, label="KF")
    p2 = plt.plot(Xp[2,:], m=:o, label="Predicted", legend=:topleft, ylabel="Velocity [m/s]", xlabel="Time [s]")
    plt.plot!(p2, Xx[2,:], m=:s, label="Measured", ribbon=Δ[2])
    plt.plot!(p2, Xk[2,:], m=:p, label="KF")
    p = plt.plot(p1, p2, size=(700,350))
    return p
end

function plot_kf_smooth(Xp, Xk, Xx, Xks, Δ)
    # Xp, Xk = reduce(hcat, Xp), reduce(hcat, Xk)
    # Xx, Xks = reduce(hcat, Xx), reduce(hcat, Xks)
    p1 = plt.plot(Xp[1,:], m=:o, label="Predicted", legend=:topleft, ylabel="Position [m]", xlabel="Time [s]")
    plt.plot!(p1, Xx[1,:], m=:s, label="Measured", ribbon=Δ[1])
    plt.plot!(p1, Xk[1,:], m=:p, label="KF")
    plt.plot!(p1, Xks[1,:], m=:h, label="Smooth")
    p2 = plt.plot(Xp[2,:], m=:o, label="Predicted", legend=:topleft, ylabel="Velocity [m/s]", xlabel="Time [s]")
    plt.plot!(p2, Xx[2,:], m=:s, label="Measured", ribbon=Δ[2])
    plt.plot!(p2, Xk[2,:], m=:p, label="KF")
    plt.plot!(p2, Xks[2,:], m=:h, label="Smooth")
    p = plt.plot(p1, p2, size=(700,350))
    return p
end

## Run KFs
X, a, Δ, ΔP, Δt = get_data()
P, Q, R, F, H, B, u, m, n = define_model(X, a, Δ, ΔP, Δt)

## Unscented Kalman Filter
f(x) = F*x .+ B .* u
g(x) = H*x
skf = run_ukf(X, f, g, P, R, Q, 1e-3; filter=kf.SVDSqrt())
Xp, Xk = kf.unpack(skf, [:xp, :xk])
p1 = plot_kf(Xp, Xk, X, Δ)

## Rauch Tung Striebel Smoother
Σ, Xkm = kf.unpack_matrix(skf, [:P, :xk])
skm = run_smoothers_bts(Xkm, F, Σ)
Xkm = kf.unpack(skf, :xk)
p1m = plot_kf_smooth(Xp, Xk, X, Xkm, Δ)
