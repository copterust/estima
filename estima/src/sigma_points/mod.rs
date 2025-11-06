//! Sigma‐point generators for the Unscented Kalman Filter.

pub use self::merwe_scaled::{MerweScaled, MerweScaled as MerweScaledSigmaPoints};
pub use self::traits::{SigmaPoints, SigmaPointsInPlace, UTSigmaCount};
pub use self::weights::UTWeights;

pub mod merwe_scaled;

#[cfg(test)]
mod tests;

mod traits;
pub mod transform;
pub mod weights;

pub use transform::unscented_transform;
