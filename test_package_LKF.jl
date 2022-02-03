
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
    w = zeros(m)
    v = zeros(m)
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

## Run Kalman Filter
function run_kf(X, P, F, H, Q, R, B, u; filter=kf.CholeskySqrt())
    n = size(X, 2)
    _, _, Xx = data_input(X, n)
    k = 1
    S = kf._innovation_covariance(P, H, R)
    K = kf._kalman_gain(P, H, S)
    skf = [ kf.LinearKF(Xx[k], Xx[k], P, K, S, Xx[1], filter) ]
    while k < n
        s = kf.linear_kalman(
            Xx[k+1], skf[k].xk, skf[k].P, F, H, Q, R;
            B=B, u=u, sqrt_filter=filter,
        )
        push!(skf, s)
        k += 1
    end
    return skf
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

function run_smoothers_mbf(skf, F, H)
    m = size(skf[1].xk, 1)
    k = length(skf)
    sm = Vector{kf.MBFSmoother}(undef, k)
    ksmoother = kf.ModifiedBrysonFrazier()
    # Λ, λ = zeros(size(lsm.F)), zeros(size(lsm.F,1))
    SH = skf[k].S \ H'
    Λ, λ = SH*H, -SH*skf[k].y
    sm[k] = kf.MBFSmoother(skf[k].xk, skf[k].P, Λ, λ)
    while k > 1
        s = kf.kalman_smooth(
            ksmoother,
            skf[k-1].xk,
            skf[k-1].P,
            skf[k-1].K,
            F,
            H,
            sm[k].L,
            sm[k].l,
            skf[k-1].S,
            skf[k-1].y,
        )
        k -= 1
        sm[k] = s
    end
    return sm
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

## Simple Kalman filter
skf = run_kf(X, P, F, H, Q, R, B, u; filter=kf.CholeskyModSqrt())
Xp, Xk = kf.unpack(skf, [:xp, :xk])
p = plot_kf(Xp, Xk, X, Δ)

## Rauch Tung Striebel Smoother
Σ, Xk2m = kf.unpack_matrix(skf, [:P, :xk])
skfm = run_smoothers_bts(Xk2m, F, Σ)
Xk2m = kf.unpack(skfm, :xk)
p2m = plot_kf_smooth(Xp, Xk, X, Xk2m, Δ)

## Modified Bryson–Frazier Smoother
skfmb = run_smoothers_mbf(skf, F, H)
Xk2mb = kf.unpack(skfmb, :xk)
p2m = plot_kf_smooth(Xp, Xk, X, Xk2mb, Δ)
