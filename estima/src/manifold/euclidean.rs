//! Euclidean space manifold implementation.
//!
//! This module provides a manifold implementation for standard Euclidean spaces (R^n).
//! For Euclidean spaces, the manifold operations are trivial since the space is flat.

use super::{InitialGuess, Manifold, MeanError};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OMatrix, OVector, RealField};

/// A manifold representing n-dimensional Euclidean space (R^n).
///
/// This is the simplest manifold where:
/// - retract(x, delta) = x + delta (simple vector addition)
/// - local(x, y) = y - x (simple vector subtraction)
/// - The tangent space at any point is the same as the manifold itself
///
/// # Type Parameters
/// * `T` - Scalar type (f32 or f64)
/// * `Dim` - Dimension of the Euclidean space
///
/// # Example
/// ```rust,ignore
/// // 3D position in Euclidean space
/// let position = EuclideanManifold::new(Vector3::new(1.0, 2.0, 3.0));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct EuclideanManifold<T, Dim>
where
    Dim: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<Dim>,
{
    /// The point in Euclidean space
    pub vector: OVector<T, Dim>,
}

impl<T, Dim> EuclideanManifold<T, Dim>
where
    Dim: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<Dim>,
{
    /// Create a new Euclidean manifold point.
    ///
    /// # Arguments
    /// * `vector` - The point in Euclidean space
    pub fn new(vector: OVector<T, Dim>) -> Self {
        Self { vector }
    }

    /// Get the underlying vector.
    pub fn as_vector(&self) -> &OVector<T, Dim> {
        &self.vector
    }

    /// Get the underlying vector mutably.
    pub fn as_vector_mut(&mut self) -> &mut OVector<T, Dim> {
        &mut self.vector
    }

    /// Convert into the underlying vector.
    pub fn into_vector(self) -> OVector<T, Dim> {
        self.vector
    }
}

impl<T, Dim> Manifold<Dim, T> for EuclideanManifold<T, Dim>
where
    Dim: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<Dim> + Allocator<Dim, Dim>,
{
    fn retract(&self, delta: &OVector<T, Dim>) -> Self {
        Self::new(&self.vector + delta)
    }

    fn local(&self, other: &Self) -> OVector<T, Dim> {
        &other.vector - &self.vector
    }

    fn weighted_mean(
        points: &[Self],
        weights: &[T],
        _tolerance: T,
        _initial_guess: InitialGuess<Self>,
        _max_iterations: usize,
    ) -> Result<Self, MeanError>
    where
        DefaultAllocator: Allocator<Dim>,
    {
        if points.is_empty() || weights.is_empty() {
            return Err(MeanError::EmptyInput);
        }

        if points.len() != weights.len() {
            return Err(MeanError::LengthMismatch);
        }

        let mut weighted_sum = OVector::<T, Dim>::zeros();
        let mut total_weight = T::zero();

        for (point, &weight) in points.iter().zip(weights.iter()) {
            if weight > T::zero() {
                weighted_sum += &point.vector * weight;
                total_weight += weight;
            }
        }

        if total_weight <= T::zero() {
            return Err(MeanError::NoPositiveWeights);
        }

        let mean_vector = weighted_sum / total_weight;
        Ok(Self::new(mean_vector))
    }

    fn batch_retract(points: &[Self], deltas: &[OVector<T, Dim>], output: &mut [Self])
    where
        DefaultAllocator: Allocator<Dim>,
    {
        assert_eq!(
            points.len(),
            deltas.len(),
            "Points and deltas length mismatch"
        );
        assert_eq!(
            points.len(),
            output.len(),
            "Points and output length mismatch"
        );

        for ((point, delta), out) in points.iter().zip(deltas.iter()).zip(output.iter_mut()) {
            *out = point.retract(delta);
        }
    }

    fn batch_local(base_points: &[Self], target_points: &[Self], output: &mut [OVector<T, Dim>])
    where
        DefaultAllocator: Allocator<Dim>,
    {
        assert_eq!(
            base_points.len(),
            target_points.len(),
            "Base and target points length mismatch"
        );
        assert_eq!(
            base_points.len(),
            output.len(),
            "Base points and output length mismatch"
        );

        for ((base, target), out) in base_points
            .iter()
            .zip(target_points.iter())
            .zip(output.iter_mut())
        {
            *out = base.local(target);
        }
    }

    fn batch_local_from_base(
        base_point: &Self,
        target_points: &[Self],
        output: &mut [OVector<T, Dim>],
    ) where
        DefaultAllocator: Allocator<Dim>,
    {
        assert_eq!(
            target_points.len(),
            output.len(),
            "Target points and output length mismatch"
        );

        for (target, out) in target_points.iter().zip(output.iter_mut()) {
            *out = base_point.local(target);
        }
    }

    fn batch_local_into_matrix<C>(
        base_point: &Self,
        target_points: &[Self],
        output_matrix: &mut OMatrix<T, Dim, C>,
    ) where
        C: DimName,
        DefaultAllocator: Allocator<Dim> + Allocator<Dim, C>,
    {
        assert_eq!(
            target_points.len(),
            output_matrix.ncols(),
            "Target points length must match matrix columns"
        );

        for (i, target) in target_points.iter().enumerate() {
            let tangent = base_point.local(target);
            output_matrix.column_mut(i).copy_from(&tangent);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use nalgebra::{Vector2, Vector3, U2};

    #[test]
    fn test_euclidean_construction() {
        let point = EuclideanManifold::new(Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(point.as_vector(), &Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_euclidean_retract_local() {
        let point1 = EuclideanManifold::new(Vector2::new(1.0, 2.0));
        let point2 = EuclideanManifold::new(Vector2::new(3.0, 4.0));

        // Test local operation
        let delta = point1.local(&point2);
        assert_eq!(delta, Vector2::new(2.0, 2.0));

        // Test retract operation
        let retracted = point1.retract(&delta);
        assert_eq!(retracted.as_vector(), point2.as_vector());
    }

    #[test]
    fn test_euclidean_weighted_mean() {
        let point1 = EuclideanManifold::new(Vector2::new(0.0, 0.0));
        let point2 = EuclideanManifold::new(Vector2::new(2.0, 4.0));
        let point3 = EuclideanManifold::new(Vector2::new(4.0, 0.0));

        let points = vec![point1, point2, point3];
        let weights = vec![0.5, 0.25, 0.25];

        let mean = EuclideanManifold::<f64, U2>::weighted_mean(
            &points,
            &weights,
            1e-9,
            InitialGuess::First,
            10,
        )
        .unwrap();

        // Expected: 0.5 * [0,0] + 0.25 * [2,4] + 0.25 * [4,0] = [1.5, 1.0]
        assert!((mean.as_vector() - Vector2::new(1.5, 1.0)).norm() < 1e-10);
    }

    #[test]
    fn test_euclidean_batch_operations() {
        let points = vec![
            EuclideanManifold::new(Vector2::new(1.0, 2.0)),
            EuclideanManifold::new(Vector2::new(3.0, 4.0)),
        ];
        let deltas = vec![Vector2::new(0.5, 0.5), Vector2::new(-0.5, -0.5)];

        let mut retracted = vec![EuclideanManifold::new(Vector2::zeros()); points.len()];
        EuclideanManifold::batch_retract(&points, &deltas, &mut retracted);
        assert_eq!(retracted[0].as_vector(), &Vector2::new(1.5, 2.5));
        assert_eq!(retracted[1].as_vector(), &Vector2::new(2.5, 3.5));

        let base_point = EuclideanManifold::new(Vector2::new(0.0, 0.0));
        let mut locals = vec![Vector2::zeros(); points.len()];
        EuclideanManifold::batch_local_from_base(&base_point, &points, &mut locals);
        assert_eq!(locals[0], Vector2::new(1.0, 2.0));
        assert_eq!(locals[1], Vector2::new(3.0, 4.0));
    }
}
