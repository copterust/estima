pub mod square_root;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OVector, RealField};
pub use square_root::SquareRootUKF;

/// Holds the UT weights at compile-time Σ
#[derive(Clone, Debug)]
pub struct UTWeights<SigmaCount: DimName, T: RealField + Copy>
where
    DefaultAllocator: Allocator<SigmaCount>,
{
    /// Mean‐recombination weights
    pub w_mean: OVector<T, SigmaCount>,
    /// Covariance‐recombination weights
    pub w_covar: OVector<T, SigmaCount>,
}
