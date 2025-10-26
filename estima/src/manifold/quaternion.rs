//! Unit quaternion manifold implementation for SO(3).
//!
//! This manifold represents rotations using unit quaternions. The tangent space is the
//! 3D vector space of axis-angle increments. Retraction applies a quaternion exponential,
//! and the local map computes the logarithm of the relative rotation.

use super::{InitialGuess, Manifold, MeanError};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, OMatrix, RealField, UnitQuaternion, Vector3, U3,
};

/// Manifold wrapper around `UnitQuaternion`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitQuaternionManifold<T>
where
    T: RealField + Copy,
{
    quaternion: UnitQuaternion<T>,
}

impl<T> UnitQuaternionManifold<T>
where
    T: RealField + Copy,
{
    /// Construct from a unit quaternion.
    pub fn new(quaternion: UnitQuaternion<T>) -> Self {
        Self { quaternion }
    }

    /// Borrow the underlying quaternion.
    pub fn as_quaternion(&self) -> &UnitQuaternion<T> {
        &self.quaternion
    }

    /// Mutable access to the underlying quaternion.
    pub fn as_quaternion_mut(&mut self) -> &mut UnitQuaternion<T> {
        &mut self.quaternion
    }

    /// Consume the manifold wrapper and return the quaternion.
    pub fn into_quaternion(self) -> UnitQuaternion<T> {
        self.quaternion
    }

    /// Identity rotation.
    pub fn identity() -> Self {
        Self::new(UnitQuaternion::identity())
    }

    /// Construct from Euler angles (roll, pitch, yaw).
    pub fn from_euler_angles(roll: T, pitch: T, yaw: T) -> Self {
        Self::new(UnitQuaternion::from_euler_angles(roll, pitch, yaw))
    }
}

impl<T> Manifold<U3, T> for UnitQuaternionManifold<T>
where
    T: RealField + Copy,
    DefaultAllocator: Allocator<U3> + Allocator<U3, U3>,
{
    fn retract(&self, delta: &Vector3<T>) -> Self {
        let delta_quat = UnitQuaternion::from_scaled_axis(*delta);
        let mut result = self.quaternion * delta_quat;
        result.renormalize(); // Ensure unit length for low-precision types
        Self::new(result)
    }

    fn local(&self, other: &Self) -> Vector3<T> {
        let relative = self.quaternion.inverse() * other.quaternion;
        relative.scaled_axis()
    }

    fn weighted_mean(
        points: &[Self],
        weights: &[T],
        tolerance: T,
        initial_guess: InitialGuess<Self>,
        max_iterations: usize,
    ) -> Result<Self, MeanError>
    where
        DefaultAllocator: Allocator<U3>,
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

        let mut delta = Vector3::<T>::zeros();
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

    fn batch_retract(points: &[Self], deltas: &[Vector3<T>], output: &mut [Self])
    where
        DefaultAllocator: Allocator<U3>,
    {
        assert_eq!(points.len(), deltas.len(), "points/deltas length mismatch");
        assert_eq!(points.len(), output.len(), "points/output length mismatch");

        for ((point, delta), out) in points.iter().zip(deltas.iter()).zip(output.iter_mut()) {
            *out = point.retract(delta);
        }
    }

    fn batch_local(points_a: &[Self], points_b: &[Self], output: &mut [Vector3<T>])
    where
        DefaultAllocator: Allocator<U3>,
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

    fn batch_local_from_base(base_point: &Self, target_points: &[Self], output: &mut [Vector3<T>])
    where
        DefaultAllocator: Allocator<U3>,
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
        output_matrix: &mut OMatrix<T, U3, C>,
    ) where
        C: nalgebra::DimName,
        DefaultAllocator: Allocator<U3> + Allocator<U3, C>,
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
    use nalgebra::Vector3;

    #[test]
    fn identity_is_neutral() {
        let q = UnitQuaternionManifold::<f64>::identity();
        let delta = Vector3::zeros();
        let result = q.retract(&delta);
        assert!((q.local(&result)).norm() < 1e-12);
    }

    #[test]
    fn retract_then_local_round_trip() {
        let q = UnitQuaternionManifold::<f64>::identity();
        let delta = Vector3::new(0.1, -0.2, 0.3);
        let q2 = q.retract(&delta);
        let recovered = q.local(&q2);
        assert!((delta - recovered).norm() < 1e-10);
    }

    #[test]
    fn weighted_mean_between_rotations() {
        let q1 = UnitQuaternionManifold::<f64>::identity();
        let q2 = UnitQuaternionManifold::from_euler_angles(0.2, 0.0, 0.0);
        let q3 = UnitQuaternionManifold::from_euler_angles(0.0, 0.2, 0.0);

        let points = vec![q1, q2, q3];
        let weights = vec![0.5, 0.25, 0.25];

        let mean =
            UnitQuaternionManifold::weighted_mean(&points, &weights, 1e-9, InitialGuess::First, 50)
                .unwrap();

        let axis_angle = mean.as_quaternion().scaled_axis();
        assert!(axis_angle.norm() < 0.3);
    }

    #[test]
    fn batch_operations_behave() {
        let base = UnitQuaternionManifold::<f64>::identity();
        let other = UnitQuaternionManifold::from_euler_angles(0.1, 0.0, 0.0);
        let points = vec![base, other];
        let deltas = vec![Vector3::new(0.05, 0.0, 0.0), Vector3::new(0.0, 0.02, 0.0)];

        let mut out = vec![UnitQuaternionManifold::identity(); points.len()];
        UnitQuaternionManifold::batch_retract(&points, &deltas, &mut out);
        assert_eq!(out.len(), points.len());

        let mut locals = vec![Vector3::zeros(); points.len()];
        UnitQuaternionManifold::batch_local_from_base(&base, &points, &mut locals);
        assert!(locals[0].norm() < 1e-12);

        let mut matrix = OMatrix::<f64, U3, nalgebra::Const<2>>::zeros();
        UnitQuaternionManifold::batch_local_into_matrix(&base, &points, &mut matrix);
        assert!(matrix.column(0).norm() < 1e-12);
    }

    #[test]
    fn large_rotation_handled() {
        let q = UnitQuaternionManifold::<f64>::identity();
        let delta = Vector3::new(PI - 0.1, 0.0, 0.0);
        let q2 = q.retract(&delta);
        let recovered = q.local(&q2);
        assert!((recovered - delta).norm() < 1e-10);
    }

    #[test]
    fn weighted_mean_no_positive_weights() {
        let points = vec![UnitQuaternionManifold::identity()];
        let weights = vec![0.0];
        let result =
            UnitQuaternionManifold::weighted_mean(&points, &weights, 1e-9, InitialGuess::First, 50);
        assert_eq!(result, Err(MeanError::NoPositiveWeights));
    }

    #[test]
    fn hemisphere_sign_invariance() {
        use alloc::vec::Vec;
        // cluster of nearby rotations
        let angles = [0.20, 0.21, 0.19, 0.205, 0.195];
        let points: Vec<UnitQuaternionManifold<f64>> = angles
            .iter()
            .map(|&a| UnitQuaternionManifold::from_euler_angles(a, 0.0, 0.0))
            .collect();

        let mut points_flipped = points.clone();
        for &idx in &[0, 2, 4] {
            let raw = points_flipped[idx].as_quaternion().quaternion().clone();
            let neg = -raw;
            points_flipped[idx] = UnitQuaternionManifold::new(UnitQuaternion::new_unchecked(neg));
        }

        let weights = vec![0.5, 0.2, 0.15, 0.1, 0.05];

        // Use an initial guess that is not one of the points to avoid trivial hiding of the problem
        let init = InitialGuess::Provided(UnitQuaternionManifold::identity());

        let mean_orig = UnitQuaternionManifold::weighted_mean(&points, &weights, 1e-12, init, 200)
            .expect("mean on original should converge");

        let init = InitialGuess::Provided(UnitQuaternionManifold::identity());
        let mean_flipped =
            UnitQuaternionManifold::weighted_mean(&points_flipped, &weights, 1e-12, init, 200)
                .expect("mean on flipped should converge");

        let dot = mean_orig
            .as_quaternion()
            .quaternion()
            .coords
            .dot(&mean_flipped.as_quaternion().quaternion().coords);
        let abs_dot = dot.abs().clamp(-1.0, 1.0);
        let angle = 2.0 * abs_dot.acos();

        assert!(
            angle < 1e-12,
            "Means differ after sign flips; angular diff = {:.3e} rad (dot = {})",
            angle,
            dot
        );
    }
}
