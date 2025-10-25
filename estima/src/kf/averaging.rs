//! Weighted averaging strategies for state estimation
//!
//! This module provides traits and implementations for computing weighted means
//! of state vectors, supporting both Euclidean and non-Euclidean (manifold) spaces.

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OMatrix, OVector, RealField};

/// Trait for computing weighted averages of states.
///
/// This trait allows customization of how states are averaged, which is critical
/// for non-Euclidean spaces like rotations (SO(3)) where linear averaging is incorrect.
pub trait WeightedMean<N: DimName, T: RealField + Copy>: Clone
where
    DefaultAllocator: Allocator<N>,
{
    /// Compute the weighted mean of sigma points.
    ///
    /// # Arguments
    /// * `sigma_points` - Matrix where each column is a sigma point
    /// * `weights` - Vector of weights for each sigma point
    /// * `output` - Pre-allocated buffer for the result
    ///
    /// # Returns
    /// The weighted mean state vector
    fn weighted_mean<SigmaCount: DimName>(
        &self,
        sigma_points: &OMatrix<T, N, SigmaCount>,
        weights: &OVector<T, SigmaCount>,
        output: &mut OVector<T, N>,
    ) where
        DefaultAllocator: Allocator<N, SigmaCount> + Allocator<SigmaCount>;
}

/// Linear averaging for Euclidean spaces (default).
///
/// This implements the standard linear weighted sum:
/// mean = Σ(w_i * x_i)
#[derive(Clone, Debug)]
pub struct LinearAveraging;

impl<N: DimName, T: RealField + Copy> WeightedMean<N, T> for LinearAveraging
where
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
        // Zero out the output buffer
        output.fill(T::zero());

        // Compute linear weighted sum
        let n_sigmas = sigma_points.ncols().min(weights.len());
        for i in 0..n_sigmas {
            let weight = weights[i];
            let sigma_i = sigma_points.column(i);
            output.axpy(weight, &sigma_i, T::one());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2x3, Vector2, Vector3};

    #[test]
    fn test_linear_averaging() {
        // Each column is a sigma point: col0=[1,2], col1=[3,4], col2=[5,6]
        let sigma_points = Matrix2x3::new(1.0, 3.0, 5.0, 2.0, 4.0, 6.0);
        let weights = Vector3::new(0.2, 0.3, 0.5);
        let mut output = Vector2::zeros();

        let averaging = LinearAveraging;
        averaging.weighted_mean(&sigma_points, &weights, &mut output);

        // Actually with these columns:
        // 0.2*[1,2] + 0.3*[3,4] + 0.5*[5,6] = [0.2+0.9+2.5, 0.4+1.2+3.0] = [3.6, 4.6]
        assert!(f64::abs(output[0] - 3.6) < 1e-10);
        assert!(f64::abs(output[1] - 4.6) < 1e-10);
    }
}
