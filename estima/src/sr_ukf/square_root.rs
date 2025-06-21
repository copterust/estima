use core::marker::PhantomData;
use nalgebra::{
    allocator::Allocator, base::OMatrix, dimension::DimName, DefaultAllocator, RealField,
};

use crate::kf::traits::*;
use crate::sigma_points::SigmaPointsInPlace;
use crate::sr_ukf::UTWeights;

#[derive(Clone, Debug)]
pub struct SquareRootUKF<S, N, P, C, M, Z, SP, T>
where
    S: State<N, T>,
    N: DimName,
    P: ProcessModel<N, C, T>,
    C: DimName,
    M: MeasurementModel<N, Z, T>,
    Z: DimName,
    SP: SigmaPointsInPlace<N, T>,
    T: RealField + Copy,
    DefaultAllocator: Allocator<N>
        + Allocator<N, N>
        + Allocator<C>
        + Allocator<Z>
        + Allocator<N, <SP as SigmaPointsInPlace<N, T>>::SigmaCount>
        + Allocator<<SP as SigmaPointsInPlace<N, T>>::SigmaCount>,
{
    /// State convertible to vector
    state: S,
    /// Cholesky factor of the state covariance
    chol_cov: OMatrix<T, N, N>,
    /// Discrete-time process model
    process_model: P,
    /// Sqrt-factor of the measurement noise covariance R
    measurement_model: M,
    /// In-place sigma point generator
    sigma_point_generator: SP,
    /// Weights for UT
    weights: UTWeights<SP::SigmaCount, T>,
    /// Marker to keep the unused control‐dimension generic alive
    _phantom_c: PhantomData<C>,
    /// Marker to keep the unused measurement‐dimension generic alive
    _phantom_z: PhantomData<Z>,
}
