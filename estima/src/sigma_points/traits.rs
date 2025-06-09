use nalgebra::base::{
    allocator::Allocator,
    dimension::{DimAdd, DimMul, U1, U2},
};
use nalgebra::{DefaultAllocator, DimName, OMatrix, OVector, RealField};

/// The compile‐time number of sigma points (2 × N + 1) for the standard UT.
pub type UTSigmaCount<N> = <<N as DimMul<U2>>::Output as DimAdd<U1>>::Output;

/// Trait for sigma point generators
pub trait SigmaPoints<N, T>
where
    N: DimName,
    T: RealField,
    Self::SigmaCount: DimName,
    DefaultAllocator: Allocator<N>
        + Allocator<N, N>
        + Allocator<<Self as SigmaPoints<N, T>>::SigmaCount>
        + Allocator<N, <Self as SigmaPoints<N, T>>::SigmaCount>,
{
    /// Number of sigma points
    type SigmaCount: DimName;

    /// Generate sigma points and weights from a mean and sqrt covariance
    fn generate(
        &self,
        mean: &OVector<T, N>,
        sqrt_cov: &OMatrix<T, N, N>,
    ) -> (
        OMatrix<T, N, Self::SigmaCount>, // sigma points
        OVector<T, Self::SigmaCount>,    // mean weights
        OVector<T, Self::SigmaCount>,    // covariance weights
    );
}

/// In‐place sigma‐point generator trait
pub trait SigmaPointsInPlace<N, T>
where
    N: DimName + DimMul<U2>,
    <N as DimMul<U2>>::Output: DimAdd<U1>,
    UTSigmaCount<N>: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<N>
        + Allocator<N, N>
        + Allocator<<Self as SigmaPointsInPlace<N, T>>::SigmaCount>
        + Allocator<N, <Self as SigmaPointsInPlace<N, T>>::SigmaCount>,
{
    type SigmaCount: DimName;

    /// Generate into the provided buffers.
    ///
    /// - `sigma_pts.shape = (N, SigmaCount)`
    /// - `w_mean.len = SigmaCount`
    /// - `w_covar.len = SigmaCount`
    fn generate_into(
        &self,
        mean: &OVector<T, N>,
        sqrt_cov: &OMatrix<T, N, N>,
        sigma_pts: &mut OMatrix<T, N, Self::SigmaCount>,
        w_mean: &mut OVector<T, Self::SigmaCount>,
        w_covar: &mut OVector<T, Self::SigmaCount>,
    );
}
