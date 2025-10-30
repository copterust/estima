//! Composite manifold for combining heterogeneous state components.
//!
//! This allows constructing complex states (e.g., attitude + bias) by pairing two
//! manifolds. The tangent space is the direct sum of the component tangent spaces.

use super::{InitialGuess, Manifold, MeanError};
use alloc::vec::Vec;
use core::marker::PhantomData;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimAdd, DimName, DimSum, OMatrix, OVector, RealField,
};

/// Composite manifold of two components.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeManifold<T, M1, M2, Dim1, Dim2>
where
    M1: Manifold<Dim1, T>,
    M2: Manifold<Dim2, T>,
    Dim1: DimName + DimAdd<Dim2>,
    Dim2: DimName,
    DimSum<Dim1, Dim2>: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<Dim1> + Allocator<Dim2> + Allocator<DimSum<Dim1, Dim2>>,
{
    pub first: M1,
    pub second: M2,
    _phantom_t: PhantomData<T>,
    _phantom_dim1: PhantomData<Dim1>,
    _phantom_dim2: PhantomData<Dim2>,
}

impl<T, M1, M2, Dim1, Dim2> CompositeManifold<T, M1, M2, Dim1, Dim2>
where
    M1: Manifold<Dim1, T>,
    M2: Manifold<Dim2, T>,
    Dim1: DimName + DimAdd<Dim2>,
    Dim2: DimName,
    DimSum<Dim1, Dim2>: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<Dim1> + Allocator<Dim2> + Allocator<DimSum<Dim1, Dim2>>,
{
    /// Construct from component manifolds.
    pub fn new(first: M1, second: M2) -> Self {
        Self {
            first,
            second,
            _phantom_t: PhantomData,
            _phantom_dim1: PhantomData,
            _phantom_dim2: PhantomData,
        }
    }

    /// Decompose into components.
    pub fn into_components(self) -> (M1, M2) {
        (self.first, self.second)
    }
}

impl<T, M1, M2, Dim1, Dim2> Manifold<DimSum<Dim1, Dim2>, T>
    for CompositeManifold<T, M1, M2, Dim1, Dim2>
where
    M1: Manifold<Dim1, T>,
    M2: Manifold<Dim2, T>,
    Dim1: DimName + DimAdd<Dim2>,
    Dim2: DimName,
    DimSum<Dim1, Dim2>: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<Dim1>
        + Allocator<Dim2>
        + Allocator<DimSum<Dim1, Dim2>>
        + Allocator<nalgebra::Const<1>, Dim1>
        + Allocator<nalgebra::Const<1>, Dim2>,
{
    fn retract(&self, delta: &OVector<T, DimSum<Dim1, Dim2>>) -> Self {
        let dim1 = Dim1::dim();
        let dim2 = Dim2::dim();
        let mut delta_first = OVector::<T, Dim1>::zeros();
        for row in 0..dim1 {
            delta_first[row] = delta[(row, 0)];
        }
        let mut delta_second = OVector::<T, Dim2>::zeros();
        for row in 0..dim2 {
            delta_second[row] = delta[(dim1 + row, 0)];
        }
        let first = self.first.retract(&delta_first);
        let second = self.second.retract(&delta_second);
        Self::new(first, second)
    }

    fn local(&self, other: &Self) -> OVector<T, DimSum<Dim1, Dim2>> {
        let dim1 = Dim1::dim();
        let dim2 = Dim2::dim();
        let mut result = OVector::<T, DimSum<Dim1, Dim2>>::zeros();
        result
            .rows_mut(0, dim1)
            .copy_from(&self.first.local(&other.first));
        result
            .rows_mut(dim1, dim2)
            .copy_from(&self.second.local(&other.second));
        result
    }

    fn weighted_mean(
        points: &[Self],
        weights: &[T],
        tolerance: T,
        initial_guess: InitialGuess<Self>,
        max_iterations: usize,
    ) -> Result<Self, MeanError>
    where
        DefaultAllocator: Allocator<DimSum<Dim1, Dim2>>,
    {
        if points.is_empty() {
            return Err(MeanError::EmptyInput);
        }

        let first_points: Vec<M1> = points.iter().map(|p| p.first.clone()).collect();
        let second_points: Vec<M2> = points.iter().map(|p| p.second.clone()).collect();

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

        let mean_first = M1::weighted_mean(
            &first_points,
            weights,
            tolerance,
            first_guess,
            max_iterations,
        )?;
        let mean_second = M2::weighted_mean(
            &second_points,
            weights,
            tolerance,
            second_guess,
            max_iterations,
        )?;

        Ok(Self::new(mean_first, mean_second))
    }

    fn batch_retract(
        points: &[Self],
        deltas: &[OVector<T, DimSum<Dim1, Dim2>>],
        output: &mut [Self],
    ) where
        DefaultAllocator: Allocator<DimSum<Dim1, Dim2>>,
    {
        assert_eq!(points.len(), deltas.len(), "points/deltas length mismatch");
        assert_eq!(points.len(), output.len(), "points/output length mismatch");

        let dim1 = Dim1::dim();
        let dim2 = Dim2::dim();

        for ((point, delta), out) in points.iter().zip(deltas.iter()).zip(output.iter_mut()) {
            let mut delta_first = OVector::<T, Dim1>::zeros();
            for row in 0..dim1 {
                delta_first[row] = delta[(row, 0)];
            }
            let mut delta_second = OVector::<T, Dim2>::zeros();
            for row in 0..dim2 {
                delta_second[row] = delta[(dim1 + row, 0)];
            }
            let first = point.first.retract(&delta_first);
            let second = point.second.retract(&delta_second);
            *out = Self::new(first, second);
        }
    }

    fn batch_local(
        base_points: &[Self],
        target_points: &[Self],
        output: &mut [OVector<T, DimSum<Dim1, Dim2>>],
    ) where
        DefaultAllocator: Allocator<DimSum<Dim1, Dim2>>,
    {
        assert_eq!(
            base_points.len(),
            target_points.len(),
            "base/target length mismatch"
        );
        assert_eq!(
            base_points.len(),
            output.len(),
            "base/output length mismatch"
        );

        let dim1 = Dim1::dim();
        let dim2 = Dim2::dim();

        for ((base, target), out) in base_points
            .iter()
            .zip(target_points.iter())
            .zip(output.iter_mut())
        {
            let local_first = base.first.local(&target.first);
            let local_second = base.second.local(&target.second);
            out.rows_mut(0, dim1).copy_from(&local_first);
            out.rows_mut(dim1, dim2).copy_from(&local_second);
        }
    }

    fn batch_local_from_base(
        base_point: &Self,
        target_points: &[Self],
        output: &mut [OVector<T, DimSum<Dim1, Dim2>>],
    ) where
        DefaultAllocator: Allocator<DimSum<Dim1, Dim2>>,
    {
        assert_eq!(
            target_points.len(),
            output.len(),
            "target/output length mismatch"
        );

        let dim1 = Dim1::dim();
        let dim2 = Dim2::dim();

        for (target, out) in target_points.iter().zip(output.iter_mut()) {
            let local_first = base_point.first.local(&target.first);
            let local_second = base_point.second.local(&target.second);
            out.rows_mut(0, dim1).copy_from(&local_first);
            out.rows_mut(dim1, dim2).copy_from(&local_second);
        }
    }

    fn batch_local_into_matrix<C>(
        base_point: &Self,
        target_points: &[Self],
        output_matrix: &mut OMatrix<T, DimSum<Dim1, Dim2>, C>,
    ) where
        C: DimName,
        DefaultAllocator: Allocator<DimSum<Dim1, Dim2>> + Allocator<DimSum<Dim1, Dim2>, C>,
    {
        assert_eq!(
            target_points.len(),
            output_matrix.ncols(),
            "target length mismatch with matrix columns"
        );

        let dim1 = Dim1::dim();
        let dim2 = Dim2::dim();

        for (col_idx, target) in target_points.iter().enumerate() {
            let local_first = base_point.first.local(&target.first);
            let local_second = base_point.second.local(&target.second);
            let mut column = output_matrix.column_mut(col_idx);
            column.rows_mut(0, dim1).copy_from(&local_first);
            column.rows_mut(dim1, dim2).copy_from(&local_second);
        }
    }
}
