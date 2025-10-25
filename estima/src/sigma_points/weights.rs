//! Unscented transform weights shared by sigma-point generators and filters.

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OVector, RealField};

/// Holds the mean and covariance recombination weights.
#[derive(Clone, Debug)]
pub struct UTWeights<SigmaCount: DimName, T: RealField + Copy>
where
    DefaultAllocator: Allocator<SigmaCount>,
{
    /// Mean recombination weights.
    pub w_mean: OVector<T, SigmaCount>,
    /// Covariance recombination weights.
    pub w_covar: OVector<T, SigmaCount>,
}

impl<SigmaCount, T> UTWeights<SigmaCount, T>
where
    SigmaCount: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<SigmaCount>,
{
    /// Construct UT weights from the Merwe scaled parameters.
    pub fn from_merwe(dim: usize, alpha: T, beta: T, kappa: T) -> Self {
        let n = T::from_usize(dim).expect("dimension must fit into scalar type");
        let two = T::one() + T::one();

        let lambda = alpha * alpha * (n + kappa) - n;
        let n_lambda = n + lambda;

        let mut w_mean = OVector::<T, SigmaCount>::zeros();
        let mut w_covar = OVector::<T, SigmaCount>::zeros();

        if n_lambda.abs() < T::default_epsilon() {
            let point_count = T::from_usize(dim * 2 + 1).unwrap_or_else(|| T::one());
            let wm0 = T::one() / point_count;
            let wc0 = wm0 + (T::one() - alpha * alpha + beta);

            let remaining = T::from_usize(dim * 2).unwrap_or_else(|| two);
            let inv = if remaining.abs() < T::default_epsilon() {
                T::one() / (two * (n.abs() + T::one()))
            } else {
                T::one() / remaining
            };

            w_mean[0] = wm0;
            w_covar[0] = wc0;

            for i in 1..SigmaCount::dim() {
                w_mean[i] = inv;
                w_covar[i] = inv;
            }
        } else {
            let wm0 = lambda / n_lambda;
            let wc0 = lambda / n_lambda + (T::one() - alpha * alpha + beta);
            let inv = T::one() / (two * n_lambda);

            w_mean[0] = wm0;
            w_covar[0] = wc0;

            for i in 1..SigmaCount::dim() {
                w_mean[i] = inv;
                w_covar[i] = inv;
            }
        }

        Self { w_mean, w_covar }
    }
}
