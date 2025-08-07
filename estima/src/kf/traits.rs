use nalgebra::Const;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, Matrix, OVector, RealField, Storage,
};

/// A user‐provided state type must be convertible to/from an `OVector<T,N>`.
pub trait State<N: nalgebra::Dim + nalgebra::DimName, T: RealField + Copy>: Clone
where
    DefaultAllocator: Allocator<N>,
{
    /// Turn `self` into the filter's own vector form
    fn into_vector(self) -> OVector<T, N>;

    /// Write the state vector into a pre-allocated buffer without cloning
    /// Default implementation still uses into_vector, but implementations
    /// should override this for efficiency
    fn write_into(&self, buf: &mut OVector<T, N>) {
        *buf = self.clone().into_vector();
    }

    /// Convert back from the filter's vector to your application‐level state
    fn from_vector(v: OVector<T, N>) -> Self;
}

pub trait ProcessModel<N: DimName, C: DimName, T: RealField + Copy>
where
    DefaultAllocator: Allocator<N> + Allocator<C>,
{
    fn predict<S>(
        &self,
        state: &Matrix<T, N, Const<1>, S>,
        dt: T,
        control: Option<&OVector<T, C>>,
    ) -> OVector<T, N>
    where
        S: Storage<T, N, Const<1>>;
}

/// A measurement model for an N‐dimensional state and Z‐dimensional measurement.
pub trait MeasurementModel<N: DimName, Z: DimName, T: RealField + Copy>
where
    DefaultAllocator: Allocator<N> + Allocator<Z>,
{
    /// Project a state `x ∈ ℝⁿ` into measurement space `z ∈ ℝᵐ`.
    fn measure(&self, x: &OVector<T, N>) -> OVector<T, Z>;

    /// Compute the "residual" between a predicted measurement `z_pred` and an
    /// actual measurement `z_meas`.  Often just `z_meas – z_pred`, but can
    /// wrap angles, handle saturations, etc.
    fn residual(&self, z_pred: &OVector<T, Z>, z_meas: &OVector<T, Z>) -> OVector<T, Z>;
}
