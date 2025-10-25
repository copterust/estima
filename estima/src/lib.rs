#![no_std]
#![doc = "Unscented Kalman filtering primitives for `no_std` environments."]

extern crate alloc;

pub mod manifold;
pub mod sigma_points;
pub mod ukf;

pub use sigma_points::UTWeights;
pub use ukf::error::UkfError;
pub use ukf::unscented::UnscentedKalmanFilter;
