use nalgebra::{
    allocator::Allocator,
    base::{OMatrix, OVector},
    DefaultAllocator, DimMin, DimName, RealField,
};

pub fn unscented_transform<N, E, T>(
    sigma_pts: &OMatrix<T, N, E>, // N rows, E columns
    w_mean: &OVector<T, E>,
    w_covar: &OVector<T, E>,
) -> (OVector<T, N>, OMatrix<T, N, N>)
where
    N: DimName,
    E: DimName + DimMin<N, Output = N>,
    <E as DimMin<N>>::Output: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<<E as DimMin<N>>::Output>
        + Allocator<N>
        + Allocator<E>
        + Allocator<N, E>
        + Allocator<E, N>
        + Allocator<<E as DimMin<N>>::Output, N>
        + Allocator<N, <E as DimMin<N>>::Output>
        + Allocator<N, N>,
{
    let mut mean = OVector::<T, N>::zeros();
    for i in 0..E::dim() {
        mean.axpy(w_mean[i], &sigma_pts.column(i), T::one());
    }

    let mut dev = OMatrix::<T, N, E>::zeros();
    for i in 0..E::dim() {
        let f = w_covar[i].sqrt();
        let mut col = dev.column_mut(i);
        col.copy_from(&(sigma_pts.column(i) - &mean));
        col.scale_mut(f);
    }

    let qr = dev.transpose().qr();
    let r_mat = qr.r();

    let r_t = r_mat.transpose();

    (mean, r_t)
}
