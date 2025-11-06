//! S^2 manifold (unit sphere in R^3) implementation.
//!
//! This manifold represents points on the surface of a unit sphere. The tangent space
//! at a point is the 2D plane orthogonal to the point's vector representation.

use super::{InitialGuess, Manifold, MeanError};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, OMatrix, RealField, Unit, Vector2, Vector3, U2, U3,
};

/// Manifold for S^2, the unit sphere in R^3.
///
/// Points are represented by unit vectors in 3D. The tangent space is 2-dimensional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct S2Manifold<T>
where
    T: RealField + Copy,
{
    point: Unit<Vector3<T>>,
}

impl<T> S2Manifold<T>
where
    T: RealField + Copy,
{
    /// Construct from a unit vector.
    pub fn new(point: Unit<Vector3<T>>) -> Self {
        Self { point }
    }

    /// Construct from a vector, which will be normalized.
    pub fn from_vector(vector: Vector3<T>) -> Option<Self> {
        Unit::try_new(vector, T::default_epsilon()).map(|point| Self { point })
    }

    /// Borrow the underlying unit vector.
    pub fn as_unit_vector(&self) -> &Unit<Vector3<T>> {
        &self.point
    }

    /// Mutable access to the underlying unit vector.
    pub fn as_unit_vector_mut(&mut self) -> &mut Unit<Vector3<T>> {
        &mut self.point
    }

    /// Consume the manifold wrapper and return the unit vector.
    pub fn into_unit_vector(self) -> Unit<Vector3<T>> {
        self.point
    }

    /// Identity element (e.g., north pole).
    pub fn identity() -> Self {
        Self::new(Vector3::z_axis())
    }
}

/// Create a local orthonormal basis for the tangent plane at a point `p` on the sphere.
fn local_basis<T>(p: &Vector3<T>) -> (Vector3<T>, Vector3<T>)
where
    T: RealField + Copy,
{
    // Choose a vector `a` that is not collinear with `p`.
    // If `p` is mostly aligned with x-axis, choose y-axis to be `a`. Otherwise, choose x-axis.
    // This avoids issues near poles of the chosen axis `a`.
    // TODO remove unwrap
    let a = if p.x.abs() > T::from_f64(0.9).unwrap() {
        Vector3::y()
    } else {
        Vector3::x()
    };
    let e1 = p.cross(&a).normalize();
    let e2 = p.cross(&e1); // Already normalized since p and e1 are orthogonal and unit.
    (e1, e2)
}

impl<T> Manifold<U2, T> for S2Manifold<T>
where
    T: RealField + Copy,
    DefaultAllocator: Allocator<U2> + Allocator<U2, U2> + Allocator<U3>,
{
    fn retract(&self, delta: &Vector2<T>) -> Self {
        let (e1, e2) = local_basis(self.point.as_ref());
        let delta_v = e1 * delta.x + e2 * delta.y;

        let theta = delta.norm();
        if theta < T::default_epsilon() {
            return *self;
        }

        let new_point_vec = self.point.as_ref() * theta.cos() + delta_v / theta * theta.sin();
        // Renormalize for numerical stability, especially with single precision floats.
        Self::new(Unit::new_normalize(new_point_vec))
    }

    fn local(&self, other: &Self) -> Vector2<T> {
        let p1 = self.point.as_ref();
        let p2 = other.point.as_ref();

        let dot = p1
            .dot(p2)
            .clamp(T::from_f64(-1.0).unwrap(), T::from_f64(1.0).unwrap());
        let theta = dot.acos();

        if theta < T::default_epsilon() {
            return Vector2::zeros();
        }

        let v_unscaled = p2 - p1 * dot;
        let v_unscaled_norm = v_unscaled.norm();

        if v_unscaled_norm < T::default_epsilon() {
            // This happens for identical or antipodal points.
            // Identical points are handled by the theta check above.
            // So this is for antipodal points.
            // The logarithm is not unique. Pick an arbitrary valid tangent.
            return Vector2::new(theta, T::zero());
        }

        let v = v_unscaled / v_unscaled_norm * theta;

        let (e1, e2) = local_basis(p1);
        Vector2::new(v.dot(&e1), v.dot(&e2))
    }

    fn weighted_mean(
        points: &[Self],
        weights: &[T],
        tolerance: T,
        initial_guess: InitialGuess<Self>,
        max_iterations: usize,
    ) -> Result<Self, MeanError>
    where
        DefaultAllocator: Allocator<U2>,
    {
        if points.is_empty() || weights.is_empty() {
            return Err(MeanError::EmptyInput);
        }
        if points.len() != weights.len() {
            return Err(MeanError::LengthMismatch);
        }
        if tolerance < T::zero() {
            return Err(MeanError::InvalidTolerance);
        }
        if max_iterations == 0 {
            return Err(MeanError::NotConverged);
        }

        let mut mean = match initial_guess {
            InitialGuess::First => points[0],
            InitialGuess::Index(idx) => {
                if idx >= points.len() {
                    return Err(MeanError::IndexOutOfBounds);
                }
                points[idx]
            }
            InitialGuess::MaxWeight => {
                let mut best_idx = None;
                let mut best_weight = T::zero();
                for (i, &w) in weights.iter().enumerate() {
                    if w > T::zero() && (best_idx.is_none() || w > best_weight) {
                        best_idx = Some(i);
                        best_weight = w;
                    }
                }
                match best_idx {
                    Some(i) => points[i],
                    None => return Err(MeanError::NoPositiveWeights),
                }
            }
            InitialGuess::Provided(m) => m,
        };

        let tolerance_sq = tolerance * tolerance;

        let mut delta = Vector2::<T>::zeros();
        for _ in 0..max_iterations {
            delta.fill(T::zero());
            let mut total_weight = T::zero();

            for (point, &weight) in points.iter().zip(weights.iter()) {
                if weight > T::zero() {
                    delta += mean.local(point) * weight;
                    total_weight += weight;
                }
            }

            if total_weight <= T::zero() {
                return Err(MeanError::NoPositiveWeights);
            }

            delta /= total_weight;

            if delta.dot(&delta) <= tolerance_sq {
                return Ok(mean);
            }

            mean = mean.retract(&delta);
        }

        Err(MeanError::NotConverged)
    }

    fn batch_retract(points: &[Self], deltas: &[Vector2<T>], output: &mut [Self])
    where
        DefaultAllocator: Allocator<U2>,
    {
        assert_eq!(points.len(), deltas.len(), "points/deltas length mismatch");
        assert_eq!(points.len(), output.len(), "points/output length mismatch");

        for ((point, delta), out) in points.iter().zip(deltas.iter()).zip(output.iter_mut()) {
            *out = point.retract(delta);
        }
    }

    fn batch_local(points_a: &[Self], points_b: &[Self], output: &mut [Vector2<T>])
    where
        DefaultAllocator: Allocator<U2>,
    {
        assert_eq!(
            points_a.len(),
            points_b.len(),
            "base/target length mismatch"
        );
        assert_eq!(points_a.len(), output.len(), "base/output length mismatch");

        for ((base, target), out) in points_a.iter().zip(points_b.iter()).zip(output.iter_mut()) {
            *out = base.local(target);
        }
    }

    fn batch_local_from_base(base_point: &Self, target_points: &[Self], output: &mut [Vector2<T>])
    where
        DefaultAllocator: Allocator<U2>,
    {
        assert_eq!(
            target_points.len(),
            output.len(),
            "target/output length mismatch"
        );

        for (target, out) in target_points.iter().zip(output.iter_mut()) {
            *out = base_point.local(target);
        }
    }

    fn batch_local_into_matrix<C>(
        base_point: &Self,
        target_points: &[Self],
        output_matrix: &mut OMatrix<T, U2, C>,
    ) where
        C: nalgebra::DimName,
        DefaultAllocator: Allocator<U2> + Allocator<U2, C>,
    {
        assert_eq!(
            target_points.len(),
            output_matrix.ncols(),
            "target length mismatch with matrix columns"
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
    use core::f64::consts::PI;
    use nalgebra::{Unit, Vector3};

    #[test]
    fn test_s2_construction() {
        let p = Vector3::new(1.0, 0.0, 0.0);
        let m = S2Manifold::<f64>::from_vector(p).unwrap();
        assert!((m.as_unit_vector().into_inner() - p).norm() < 1e-9);
    }

    #[test]
    fn test_s2_retract_local_roundtrip() {
        let p1 = S2Manifold::<f64>::new(Vector3::z_axis());
        let delta = Vector2::new(0.1, -0.2);
        let p2 = p1.retract(&delta);
        let recovered_delta = p1.local(&p2);
        assert!((delta - recovered_delta).norm() < 1e-10);
    }

    #[test]
    fn test_s2_retract_local_roundtrip_large() {
        let p1 = S2Manifold::<f64>::new(Vector3::z_axis());
        let delta = Vector2::new(PI / 2.0, 0.0);
        let p2 = p1.retract(&delta);

        assert!((p2.as_unit_vector().into_inner() - Vector3::y()).norm() < 1e-10);

        let recovered_delta = p1.local(&p2);
        assert!((delta - recovered_delta).norm() < 1e-10);
    }

    #[test]
    fn test_s2_local_antipodal() {
        let p1 = S2Manifold::<f64>::new(Vector3::z_axis());
        let p2 = S2Manifold::<f64>::new(-Vector3::z_axis());
        let delta = p1.local(&p2);
        assert!((delta.norm() - PI).abs() < 1e-10);
    }

    #[test]
    fn test_s2_weighted_mean() {
        let p1 = S2Manifold::<f64>::new(Unit::new_normalize(Vector3::new(1.0, 0.1, 0.0)));
        let p2 = S2Manifold::<f64>::new(Unit::new_normalize(Vector3::new(1.0, -0.1, 0.0)));
        let points = vec![p1, p2];
        let weights = vec![0.5, 0.5];
        let mean =
            S2Manifold::weighted_mean(&points, &weights, 1e-9, InitialGuess::First, 10).unwrap();
        // Expected mean is on the x-axis
        assert!((mean.as_unit_vector().into_inner() - Vector3::x()).norm() < 1e-9);
    }

    #[test]
    fn test_s2_batch_operations() {
        let p1 = S2Manifold::<f64>::new(Vector3::z_axis());
        let p2 = S2Manifold::<f64>::new(Vector3::x_axis());
        let points = vec![p1, p2];
        let deltas = vec![Vector2::new(0.1, 0.0), Vector2::new(0.0, 0.2)];

        let mut retracted = vec![S2Manifold::identity(); 2];
        S2Manifold::<f64>::batch_retract(&points, &deltas, &mut retracted);
        assert_eq!(retracted.len(), 2);

        let mut locals = vec![Vector2::zeros(); 2];
        S2Manifold::batch_local_from_base(&p1, &points, &mut locals);
        assert!(locals[0].norm() < 1e-9);
        assert!((locals[1].norm() - PI / 2.0).abs() < 1e-9);
    }
}
