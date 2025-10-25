//! Weighted averaging strategies for sigma points.

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OMatrix, OVector, RealField};

/// Trait for computing weighted averages of sigma points.
pub trait WeightedMean<N: DimName, T: RealField + Copy>: Clone
where
    DefaultAllocator: Allocator<N>,
{
    /// Compute the weighted mean of sigma points.
    fn weighted_mean<SigmaCount: DimName>(
        &self,
        sigma_points: &OMatrix<T, N, SigmaCount>,
        weights: &OVector<T, SigmaCount>,
        output: &mut OVector<T, N>,
    ) where
        DefaultAllocator: Allocator<N, SigmaCount> + Allocator<SigmaCount>;
}

/// Linear averaging for Euclidean spaces (default).
#[derive(Clone, Debug)]
pub struct LinearAveraging;

impl<N, T> WeightedMean<N, T> for LinearAveraging
where
    N: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<N>,
{
    fn weighted_mean<SigmaCount: DimName>(
        &self,
        sigma_points: &OMatrix<T, N, SigmaCount>,
        weights: &OVector<T, SigmaCount>,
        output: &mut OVector<T, N>,
    ) where
        DefaultAllocator: Allocator<N, SigmaCount> + Allocator<SigmaCount>,
    {
        output.fill(T::zero());

        let n_sigmas = sigma_points.ncols().min(weights.len());
        for i in 0..n_sigmas {
            let weight = weights[i];
            output.axpy(weight, &sigma_points.column(i), T::one());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2x3, Vector2, Vector3};

    #[test]
    fn test_linear_averaging() {
        let sigma_points = Matrix2x3::<f64>::new(1.0, 3.0, 5.0, 2.0, 4.0, 6.0);
        let weights = Vector3::<f64>::new(0.2, 0.3, 0.5);
        let mut output = Vector2::<f64>::zeros();

        LinearAveraging.weighted_mean(&sigma_points, &weights, &mut output);

        assert!((output[0] - 3.6).abs() < 1e-12_f64);
        assert!((output[1] - 4.6).abs() < 1e-12_f64);
    }
}
