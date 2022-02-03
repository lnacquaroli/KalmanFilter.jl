module KalmanFilter

#
# References:
# Haykin, Kalman Filter and NN (2001)
# https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers
# Hartikainen et al., Optimal filtering with Kalman filters and smoothers - A Manual for Matlab toolbox EKF/UKF (2011)
# Martin, Modified Bryson-Frazier Smoother Cross-Covariance. IEEE TRANSACTIONS ON AUTOMATIC CONTROL, VOL. 59, NO. 1, JANUARY 2014
#

using LinearAlgebra

include("CommonStructures.jl")
export CholeskySqrt, SVDSqrt, CholeskyModSqrt


# Review the use of eye, because it could be replaced by I
# unpack and unpack_matrix could be merged into just unpack accepting different methods
include("Utils.jl")
export eye, unpack, unpack_matrix

include("FixedIntervalSmoothers.jl")
export kalman_smooth,
       RauchTungStriebel, ModifiedBrysonFrazier,
       RTSSmoother, MBFSmoother

include("LinearKalmanFilter.jl")
export linear_kalman, LinearKF

include("UnscentedKalmanFilter.jl")
export unscented_kalman, UnscentedKF

end # module KalmanFilter
