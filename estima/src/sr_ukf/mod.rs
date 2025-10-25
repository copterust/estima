pub mod engine;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OVector, RealField};

/// Holds the UT weights at compile-time
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

impl<SigmaCount: DimName, T: RealField + Copy> UTWeights<SigmaCount, T>
where
    DefaultAllocator: Allocator<SigmaCount>,
{
    pub fn new(n_dim: usize, alpha: T, beta: T, kappa: T) -> Self {
        let n = T::from_usize(n_dim).unwrap();
        let lambda = alpha * alpha * (n + kappa) - n;
        let n_lambda = n + lambda;

        let mut w_mean = OVector::<T, SigmaCount>::zeros();
        let mut w_covar = OVector::<T, SigmaCount>::zeros();

        let (wm0, wc0, inv_two_c) = if n_lambda.abs() < T::default_epsilon() {
            // Handle degenerate case
            let inv_2n = T::one() / (T::one() + T::one()) / n;
            (
                T::zero(),
                T::zero() + (T::one() - alpha * alpha + beta),
                inv_2n,
            )
        } else {
            (
                lambda / n_lambda,
                lambda / n_lambda + (T::one() - alpha * alpha + beta),
                T::one() / (T::one() + T::one()) / n_lambda,
            )
        };

        w_mean[0] = wm0;
        w_covar[0] = wc0;

        for i in 1..SigmaCount::dim() {
            w_mean[i] = inv_two_c;
            w_covar[i] = inv_two_c;
        }
        Self { w_mean, w_covar }
    }
}
