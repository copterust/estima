use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimAdd, DimName, DimSum, Matrix4, OVector, RealField,
    SymmetricEigen, U3, U4,
};

use crate::manifold::composite::CompositeManifold;
use crate::manifold::quaternion::UnitQuaternionManifold;
use crate::manifold::{InitialGuess, Manifold, MeanError};

/// Trait for computing the weighted mean of points on a manifold.
///
/// This allows customizing the algorithm used for mean calculation,
/// e.g., using a closed-form solution for Euclidean spaces or
/// an iterative Fréchet mean for general manifolds.
pub trait ManifoldWeightedMean<M, TangentDim, T>: Clone
where
    M: Manifold<TangentDim, T>,
    TangentDim: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<TangentDim>,
{
    /// Compute the weighted mean.
    fn compute_mean<'a, I>(
        &self,
        points: I,
        weights: &[T],
        initial_guess: InitialGuess<M>,
    ) -> Result<M, MeanError>
    where
        M: 'a,
        I: IntoIterator<Item = &'a M> + Clone,
        I::IntoIter: Clone;
}

/// Iterative Fréchet mean algorithm.
///
/// This is the general algorithm applicable to any Riemannian manifold.
/// It iteratively minimizes the sum of squared distances.
#[derive(Clone, Debug)]
pub struct FrechetMean<T> {
    pub tolerance: T,
    pub max_iterations: usize,
}

impl<T: RealField + Copy> Default for FrechetMean<T> {
    fn default() -> Self {
        Self {
            tolerance: T::from_subset(&1e-9),
            max_iterations: 100,
        }
    }
}

impl<M, TangentDim, T> ManifoldWeightedMean<M, TangentDim, T> for FrechetMean<T>
where
    M: Manifold<TangentDim, T>,
    TangentDim: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<TangentDim>,
{
    fn compute_mean<'a, I>(
        &self,
        points: I,
        weights: &[T],
        initial_guess: InitialGuess<M>,
    ) -> Result<M, MeanError>
    where
        M: 'a,
        I: IntoIterator<Item = &'a M> + Clone,
        I::IntoIter: Clone,
    {
        M::weighted_mean(
            points,
            weights,
            self.tolerance,
            initial_guess,
            self.max_iterations,
        )
    }
}

/// Closed-form mean for Euclidean-like manifolds.
///
/// This assumes the manifold has a global chart where linear averaging works directly,
/// or that the `retract` and `local` operations are linear (e.g. vector space).
///
/// **Warning**: Only use this for `EuclideanManifold` or similar flat spaces.
#[derive(Clone, Debug, Default)]
pub struct EuclideanMean;

impl<M, TangentDim, T> ManifoldWeightedMean<M, TangentDim, T> for EuclideanMean
where
    M: Manifold<TangentDim, T>,
    TangentDim: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<TangentDim>,
{
    fn compute_mean<'a, I>(
        &self,
        points: I,
        weights: &[T],
        _initial_guess: InitialGuess<M>,
    ) -> Result<M, MeanError>
    where
        M: 'a,
        I: IntoIterator<Item = &'a M> + Clone,
        I::IntoIter: Clone,
    {
        if weights.is_empty() {
            return Err(MeanError::EmptyInput);
        }

        // We assume we can just linear average in the tangent space around zero (or any point).
        // For EuclideanManifold, local(0, x) = x.
        // So we can just average the "vectors" directly if we could access them.
        // Since we only have the Manifold interface, we can pick the first point as base,
        // average the tangent vectors, and retract.
        // Or, if we trust the user, we can assume the manifold IS a vector space.

        // Let's do the safe "tangent space average" which is exact for Euclidean:
        // Mean = p0 + sum(w_i * (p_i - p0))
        // This is one step of Frechet mean, which converges instantly for Euclidean.

        let base = match points.clone().into_iter().next() {
            Some(p) => p,
            None => return Err(MeanError::EmptyInput),
        };
        let mut delta_sum = OVector::<T, TangentDim>::zeros();
        let mut total_weight = T::zero();

        for (point, &weight) in points.into_iter().zip(weights.iter()) {
            if weight > T::zero() {
                let tangent = base.local(point);
                delta_sum += tangent * weight;
                total_weight += weight;
            }
        }

        if total_weight <= T::zero() {
            return Err(MeanError::NoPositiveWeights);
        }

        delta_sum /= total_weight;
        Ok(base.retract(&delta_sum))
    }
}

/// Non-iterative "Chordal L2" mean for Unit Quaternions.
///
/// This computes the eigenvector corresponding to the largest eigenvalue of
/// M = sum(w_i * q_i * q_i^T). This minimizes the weighted sum of chordal distances
/// (squared Euclidean distance in R^4).
///
/// This is generally faster and more robust than Fréchet mean for quaternions,
/// though the resulting mean is slightly different (but usually very close).
#[derive(Clone, Debug, Default)]
pub struct ChordalMean;

impl<T> ManifoldWeightedMean<UnitQuaternionManifold<T>, U3, T> for ChordalMean
where
    T: RealField + Copy,
    DefaultAllocator: Allocator<U3> + Allocator<U4, U4> + Allocator<U4>,
{
    fn compute_mean<'a, I>(
        &self,
        points: I,
        weights: &[T],
        _initial_guess: InitialGuess<UnitQuaternionManifold<T>>,
    ) -> Result<UnitQuaternionManifold<T>, MeanError>
    where
        UnitQuaternionManifold<T>: 'a,
        I: IntoIterator<Item = &'a UnitQuaternionManifold<T>> + Clone,
        I::IntoIter: Clone,
    {
        if weights.is_empty() {
            return Err(MeanError::EmptyInput);
        }

        let mut m = Matrix4::<T>::zeros();
        let mut total_weight = T::zero();

        for (point, &weight) in points.into_iter().zip(weights.iter()) {
            if weight > T::zero() {
                let q = &point.as_quaternion().quaternion().coords;
                // M += w * q * q^T
                // Use rank-1 update if possible, or just optimized loop
                // Since 4x4 is small, manual loop is fine but let's make it cleaner
                // m += (q * q.transpose()) * weight;
                m.ger(weight, q, q, T::one());
                total_weight += weight;
            }
        }

        if total_weight <= T::zero() {
            return Err(MeanError::NoPositiveWeights);
        }

        // Symmetric eigen decomposition
        let eigen = SymmetricEigen::new(m);

        // Find index of largest eigenvalue
        let (max_eigen_idx, _) = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .unwrap_or((0, &T::zero()));

        let best_quat_vec = eigen.eigenvectors.column(max_eigen_idx).into_owned();
        let quat = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::from_vector(
            best_quat_vec,
        ));

        Ok(UnitQuaternionManifold::new(quat))
    }
}

/// Strategy for composite manifolds that delegates to component strategies.
#[derive(Clone, Debug)]
pub struct CompositeStrategy<S1, S2> {
    pub first: S1,
    pub second: S2,
}

impl<S1, S2> CompositeStrategy<S1, S2> {
    pub fn new(first: S1, second: S2) -> Self {
        Self { first, second }
    }
}

impl<S1: Default, S2: Default> Default for CompositeStrategy<S1, S2> {
    fn default() -> Self {
        Self {
            first: S1::default(),
            second: S2::default(),
        }
    }
}

impl<T, M1, M2, Dim1, Dim2, S1, S2>
    ManifoldWeightedMean<CompositeManifold<T, M1, M2, Dim1, Dim2>, DimSum<Dim1, Dim2>, T>
    for CompositeStrategy<S1, S2>
where
    T: RealField + Copy,
    M1: Manifold<Dim1, T>,
    M2: Manifold<Dim2, T>,
    Dim1: DimName + DimAdd<Dim2>,
    Dim2: DimName,
    DimSum<Dim1, Dim2>: DimName,
    S1: ManifoldWeightedMean<M1, Dim1, T>,
    S2: ManifoldWeightedMean<M2, Dim2, T>,
    DefaultAllocator: Allocator<Dim1>
        + Allocator<Dim2>
        + Allocator<DimSum<Dim1, Dim2>>
        + Allocator<nalgebra::Const<1>, Dim1>
        + Allocator<nalgebra::Const<1>, Dim2>,
{
    fn compute_mean<'a, I>(
        &self,
        points: I,
        weights: &[T],
        initial_guess: InitialGuess<CompositeManifold<T, M1, M2, Dim1, Dim2>>,
    ) -> Result<CompositeManifold<T, M1, M2, Dim1, Dim2>, MeanError>
    where
        CompositeManifold<T, M1, M2, Dim1, Dim2>: 'a,
        I: IntoIterator<Item = &'a CompositeManifold<T, M1, M2, Dim1, Dim2>> + Clone,
        I::IntoIter: Clone,
    {
        if weights.is_empty() {
            return Err(MeanError::EmptyInput);
        }

        // Create iterators for sub-components without allocating new vectors
        // We need to clone the iterator to pass it to both strategies
        // The iterator yields references to CompositeManifold, we map to references to components
        let points_clone = points.clone();
        let first_points = points.into_iter().map(|p| &p.first);
        let second_points = points_clone.into_iter().map(|p| &p.second);

        let first_guess = match &initial_guess {
            InitialGuess::First => InitialGuess::First,
            InitialGuess::MaxWeight => InitialGuess::MaxWeight,
            InitialGuess::Index(i) => InitialGuess::Index(*i),
            InitialGuess::Provided(p) => InitialGuess::Provided(p.first.clone()),
        };
        let second_guess = match &initial_guess {
            InitialGuess::First => InitialGuess::First,
            InitialGuess::MaxWeight => InitialGuess::MaxWeight,
            InitialGuess::Index(i) => InitialGuess::Index(*i),
            InitialGuess::Provided(p) => InitialGuess::Provided(p.second.clone()),
        };

        let mean_first = self
            .first
            .compute_mean(first_points, weights, first_guess)?;
        let mean_second = self
            .second
            .compute_mean(second_points, weights, second_guess)?;

        Ok(CompositeManifold::new(mean_first, mean_second))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use nalgebra::UnitQuaternion;

    #[test]
    fn chordal_mean_basic() {
        let q1 = UnitQuaternionManifold::new(UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0));
        let q2 = UnitQuaternionManifold::new(UnitQuaternion::from_euler_angles(0.2, 0.0, 0.0));
        let points = vec![q1, q2];
        let weights = vec![0.5, 0.5];

        let mean = ChordalMean
            .compute_mean(&points, &weights, InitialGuess::First)
            .unwrap();

        let angle = mean.as_quaternion().angle_to(q1.as_quaternion());
        assert!((angle - 0.1_f64).abs() < 1e-4_f64);
    }
}
