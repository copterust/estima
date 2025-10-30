use nalgebra::base::{
    allocator::Allocator,
    dimension::{DimAdd, DimMul, U1, U2},
};
use nalgebra::{Cholesky, DefaultAllocator, DimName, OMatrix, OVector, RealField};

/// The compile‐time number of sigma points (2 × N + 1) for the standard UT.
pub type UTSigmaCount<N> = <<N as DimMul<U2>>::Output as DimAdd<U1>>::Output;

pub struct SigmaPointsGenerated<T, N, SigmaCount>
where
    T: RealField,
    N: DimName,
    SigmaCount: DimName,
    DefaultAllocator: Allocator<N> + Allocator<SigmaCount> + Allocator<N, SigmaCount>,
{
    pub sigma_points: OMatrix<T, N, SigmaCount>,
    pub mean_weights: OVector<T, SigmaCount>,
    pub covariance_weights: OVector<T, SigmaCount>,
}

/// Trait for sigma point generators
pub trait SigmaPoints<N: DimName, T: RealField>
where
    T: RealField,
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
        sqrt_cov: &Cholesky<T, N>,
    ) -> SigmaPointsGenerated<T, N, <Self as SigmaPoints<N, T>>::SigmaCount>;
}

/// In‐place sigma‐point generator trait
pub trait SigmaPointsInPlace<N: DimName, T: RealField + Copy>
where
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
        sqrt_cov: &Cholesky<T, N>,
        sigma_pts: &mut OMatrix<T, N, Self::SigmaCount>,
        w_mean: &mut OVector<T, Self::SigmaCount>,
        w_covar: &mut OVector<T, Self::SigmaCount>,
    );
}
