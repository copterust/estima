//! Manifold-based state estimation support
//!
//! This module provides traits and utilities for working with states that live on manifolds.
//! A manifold is a space that locally resembles Euclidean space but may have different
//! global properties (e.g., rotations on SO(3), poses on SE(3)).

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OMatrix, OVector, RealField};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanError {
    EmptyInput,
    NoPositiveWeights,
    NotConverged,
    InvalidTolerance,
    IndexOutOfBounds,
}

pub enum InitialGuess<M> {
    First,
    MaxWeight,
    Index(usize),
    Provided(M),
}

/// A convenience helper: choose initial index (max positive weight) or fallback to 0.
#[inline]
fn choose_initial_index<T: RealField + Copy>(
    weights: &[T],
    max_index_exclusive: usize,
) -> Option<usize> {
    use core::cmp::min;
    let limit = min(max_index_exclusive, weights.len());
    let mut best: Option<(usize, T)> = None;
    let zero = T::zero();
    for (i, &w) in weights.iter().take(limit).enumerate() {
        // ignore NaN / non-positive weights because (w > zero) is false in those cases
        if w > zero {
            match best {
                None => best = Some((i, w)),
                Some((_, best_w)) if w > best_w => best = Some((i, w)),
                _ => {}
            }
        }
    }
    best.map(|(i, _)| i)
}

/// A state type that lives on a manifold with a tangent space of dimension `TangentDim`.
///
/// The manifold trait provides operations for mapping between the manifold and its tangent space:
/// - `retract`: Maps from tangent space to manifold (exponential map)
/// - `local`: Maps from manifold to tangent space (logarithmic map)
///
/// # Properties
///
/// Implementations must satisfy:
/// 1. `retract(x, local(x, y)) ≈ y` for all `x`, `y` on the manifold
/// 2. `local(x, retract(x, delta)) ≈ delta` for all `x` on the manifold and small `delta`
/// 3. `local(x, x) = 0` for all `x` on the manifold
/// 4. `retract(x, 0) = x` for all `x` on the manifold
pub trait Manifold<TangentDim: DimName, T: RealField + Copy>: Clone + Sized
where
    DefaultAllocator: Allocator<TangentDim>,
{
    /// Apply a tangent vector to this manifold point to get a new point.
    ///
    /// This is the exponential map from the tangent space at `self` to the manifold.
    ///
    /// # Arguments
    /// * `delta` - A tangent vector to apply
    ///
    /// # Returns
    /// A new point on the manifold
    fn retract(&self, delta: &OVector<T, TangentDim>) -> Self;

    /// Compute the tangent vector from this point to another point.
    ///
    /// This is the logarithmic map from the manifold to the tangent space at `self`.
    ///
    /// # Arguments
    /// * `other` - The target point on the manifold
    ///
    /// # Returns
    /// The tangent vector that maps from `self` to `other` via retract
    fn local(&self, other: &Self) -> OVector<T, TangentDim>;

    /// Compute a weighted mean of manifold points.
    ///
    /// This method computes the weighted Fréchet mean on the manifold, which generalizes
    /// the concept of weighted average to non-Euclidean spaces.
    ///
    /// # Arguments
    /// * `points` - Points on the manifold to average
    /// * `weights` - Weights for each point (should sum to 1.0)
    ///
    /// # Returns
    /// The weighted mean point on the manifold
    ///
    /// # Default Implementation
    ///
    /// The default implementation uses an iterative algorithm that:
    /// 1. Starts with the first point as initial guess
    /// 2. Iteratively moves toward the weighted mean in tangent space
    /// 3. Converges when the update is sufficiently small
    fn weighted_mean(
        points: &[Self],
        weights: &[T],
        tolerance: T,
        initial_guess: InitialGuess<Self>,
        max_iterations: usize,
    ) -> Result<Self, MeanError>
    where
        Self: Manifold<TangentDim, T>,
        TangentDim: DimName,
        T: RealField + Copy,
        DefaultAllocator: Allocator<TangentDim>,
    {
        if points.is_empty() || weights.is_empty() {
            return Err(MeanError::EmptyInput);
        }

        if tolerance < T::zero() {
            return Err(MeanError::InvalidTolerance);
        }

        if max_iterations == 0 {
            return Err(MeanError::NotConverged);
        }

        // choose initial mean
        let mut mean = match initial_guess {
            InitialGuess::First => points[0].clone(),
            InitialGuess::Index(idx) => {
                if idx >= points.len() {
                    return Err(MeanError::IndexOutOfBounds);
                }
                points[idx].clone()
            }
            InitialGuess::MaxWeight => match choose_initial_index(weights, points.len()) {
                Some(idx) => points[idx].clone(),
                None => return Err(MeanError::NoPositiveWeights),
            },
            InitialGuess::Provided(m) => m,
        };

        let zero = T::zero();
        let tolerance_sq = tolerance * tolerance;

        for _iter in 0..max_iterations {
            let mut delta = OVector::<T, TangentDim>::zeros();
            let mut total_weight = T::zero();

            for (point, &weight) in points.iter().zip(weights.iter()) {
                if weight > zero {
                    // avoid allocating the scaled vector temporary
                    let mut tangent = mean.local(point);
                    tangent *= weight;
                    delta += tangent;
                    total_weight += weight;
                }
            }

            if total_weight <= zero {
                return Err(MeanError::NoPositiveWeights);
            }

            delta /= total_weight;

            let delta_sq = delta.dot(&delta);
            if delta_sq <= tolerance_sq {
                return Ok(mean); // converged
            }

            mean = mean.retract(&delta);
        }

        Err(MeanError::NotConverged)
    }

    /// Apply multiple tangent vectors to manifold points (batch retract).
    ///
    /// This is an optimized version of retract for processing multiple tangent vectors
    /// at once. The default implementation calls retract for each pair, but manifolds
    /// can override this for better performance.
    ///
    /// # Arguments
    /// * `points` - Base points on the manifold
    /// * `deltas` - Tangent vectors to apply to each point
    ///
    /// # Returns
    /// Vector of new points on the manifold
    ///
    /// # Panics
    /// Panics if `points.len() != deltas.len()`
    fn batch_retract(points: &[Self], deltas: &[OVector<T, TangentDim>], output: &mut [Self])
    where
        Self: Manifold<TangentDim, T>,
        TangentDim: DimName,
        T: RealField + Copy,
        DefaultAllocator: Allocator<TangentDim>,
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

    /// Compute multiple tangent vectors from base points to target points (batch local).
    ///
    /// This is an optimized version of local for processing multiple point pairs
    /// at once. The default implementation calls local for each pair, but manifolds
    /// can override this for better performance.
    ///
    /// # Arguments
    /// * `base_points` - Base points on the manifold
    /// * `target_points` - Target points on the manifold
    ///
    /// # Returns
    /// Vector of tangent vectors from base to target points
    ///
    /// # Panics
    /// Panics if `base_points.len() != target_points.len()`
    fn batch_local(
        base_points: &[Self],
        target_points: &[Self],
        output: &mut [OVector<T, TangentDim>],
    ) where
        Self: Manifold<TangentDim, T>,
        TangentDim: DimName,
        T: RealField + Copy,
        DefaultAllocator: Allocator<TangentDim>,
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

    /// Batch local operation from a single base point to multiple target points.
    ///
    /// This is useful when computing deviations from a mean point to multiple sigma points.
    ///
    /// # Arguments
    /// * `base_point` - Base point on the manifold
    /// * `target_points` - Target points on the manifold
    ///
    /// # Returns
    /// Vector of tangent vectors from base to each target point
    fn batch_local_from_base(
        base_point: &Self,
        target_points: &[Self],
        output: &mut [OVector<T, TangentDim>],
    ) where
        Self: Manifold<TangentDim, T>,
        TangentDim: DimName,
        T: RealField + Copy,
        DefaultAllocator: Allocator<TangentDim>,
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

    /// Write batch local results into pre-allocated matrix columns.
    ///
    /// This is a no-allocation version of batch_local_from_base that writes
    /// results directly into matrix columns.
    ///
    /// # Arguments
    /// * `base_point` - Base point on the manifold
    /// * `target_points` - Target points on the manifold
    /// * `output_matrix` - Matrix to write tangent vectors into (columns)
    ///
    /// # Panics
    /// Panics if `target_points.len() != output_matrix.ncols()`
    fn batch_local_into_matrix<C>(
        base_point: &Self,
        target_points: &[Self],
        output_matrix: &mut OMatrix<T, TangentDim, C>,
    ) where
        Self: Manifold<TangentDim, T>,
        TangentDim: DimName,
        C: DimName,
        T: RealField + Copy,
        DefaultAllocator: Allocator<TangentDim> + Allocator<TangentDim, C>,
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

/// An error state representation in the tangent space of a manifold.
///
/// This trait is used for error-state Kalman filters where the error state
/// lives in the tangent space while the nominal state lives on the manifold.
pub trait ErrorState<Dim: DimName, T: RealField + Copy>: Clone
where
    DefaultAllocator: Allocator<Dim>,
{
    /// Create a zero error state.
    fn zeros() -> Self;

    /// Convert from a vector representation to an error state.
    fn from_vector(v: OVector<T, Dim>) -> Self;

    /// Convert from an error state to a vector representation.
    fn into_vector(self) -> OVector<T, Dim>;

    /// Write the error state into a pre-allocated buffer without consuming self.
    fn as_vector(&self) -> OVector<T, Dim> {
        self.clone().into_vector()
    }
}

/// Process model for manifold states.
///
/// This trait defines a process model that operates directly on manifold states,
/// rather than on their vector representations.
pub trait ManifoldProcess<State, Control: DimName, T: RealField + Copy>
where
    DefaultAllocator: Allocator<Control>,
{
    /// Predict the next state from the current state.
    ///
    /// # Arguments
    /// * `state` - Current state on the manifold
    /// * `dt` - Time step
    /// * `control` - Optional control input
    ///
    /// # Returns
    /// The predicted next state on the manifold
    fn predict(&self, state: &State, dt: T, control: Option<&OVector<T, Control>>) -> State;
}

/// Measurement model for manifold states.
///
/// This trait defines a measurement model that maps manifold states to measurements
/// and computes residuals between predictions and observations.
pub trait ManifoldMeasurement<
    State: Manifold<TangentDim, T>,
    TangentDim: DimName,
    MeasDim: DimName,
    T: RealField + Copy,
> where
    DefaultAllocator: Allocator<MeasDim> + Allocator<TangentDim>,
{
    /// Generate a measurement from a state.
    ///
    /// # Arguments
    /// * `state` - State on the manifold
    ///
    /// # Returns
    /// The predicted measurement
    fn measure(&self, state: &State) -> OVector<T, MeasDim>;

    /// Compute the residual between predicted and actual measurements.
    ///
    /// For many measurements this is simply `actual - predicted`, but some
    /// measurements (e.g., angles) require special handling.
    ///
    /// # Arguments
    /// * `predicted` - Predicted measurement
    /// * `measured` - Actual measurement
    ///
    /// # Returns
    /// The residual vector in the measurement space
    fn residual(
        &self,
        predicted: &OVector<T, MeasDim>,
        measured: &OVector<T, MeasDim>,
    ) -> OVector<T, MeasDim>;

    /// Computes the innovation (error) between a measured value and the predicted mean.
    ///
    /// The default implementation calls `residual`, which is suitable for many cases.
    /// However, for geometrically aware filters (like AHRS), this method can be overridden
    /// to provide a more physically meaningful error vector, such as using a cross product
    /// for rotational errors.
    fn innovation(
        &self,
        measured: &OVector<T, MeasDim>,
        predicted_mean: &OVector<T, MeasDim>,
    ) -> OVector<T, MeasDim> {
        self.residual(predicted_mean, measured)
    }
}

pub mod euclidean;
