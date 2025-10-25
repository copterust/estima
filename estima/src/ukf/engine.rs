use nalgebra::{Cholesky, Const, OMatrix, OVector, RealField};

use super::averaging::WeightedMean;
use super::error::UkfError;
use crate::sigma_points::UTWeights;

/// Core UKF math operating on raw vectors.
#[derive(Clone, Debug)]
pub struct UKFEngine<N, Z, SP, T>
where
    N: 'static + nalgebra::Dim + nalgebra::DimName,
    Z: 'static + nalgebra::Dim + nalgebra::DimName,
    SP: 'static + nalgebra::Dim + nalgebra::DimName,
    T: RealField + Copy,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<N>
        + nalgebra::allocator::Allocator<Z>
        + nalgebra::allocator::Allocator<SP>
        + nalgebra::allocator::Allocator<N, SP>
        + nalgebra::allocator::Allocator<Z, SP>
        + nalgebra::allocator::Allocator<N, N>
        + nalgebra::allocator::Allocator<Z, Z>
        + nalgebra::allocator::Allocator<N, Z>
        + nalgebra::allocator::Allocator<SP, SP>
        + nalgebra::allocator::Allocator<Const<1>, Z>
        + nalgebra::allocator::Allocator<Z, N>
        + nalgebra::allocator::Allocator<Const<1>, N>
        + nalgebra::allocator::Allocator<SP, N>
        + nalgebra::allocator::Allocator<SP, Z>,
{
    pub(crate) weights: UTWeights<SP, T>,
    pub(crate) regularization_factor: T,
    pub(crate) sigma_points: OMatrix<T, N, SP>,
    pub(crate) state_deviations: OMatrix<T, N, SP>,
    pub(crate) measurement_deviations: OMatrix<T, Z, SP>,
    pub(crate) pxz_buffer: OMatrix<T, N, Z>,
    pub(crate) z_pred_buffer: OVector<T, Z>,
    pub(crate) x_buffer: OVector<T, N>,
    pub(crate) transformed: OMatrix<T, N, SP>,
    pub(crate) z_sigma_buffer: OMatrix<T, Z, SP>,
}

impl<N, Z, SP, T> UKFEngine<N, Z, SP, T>
where
    N: 'static + nalgebra::Dim + nalgebra::DimName,
    Z: 'static + nalgebra::Dim + nalgebra::DimName,
    SP: 'static + nalgebra::Dim + nalgebra::DimName,
    T: RealField + Copy,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<N>
        + nalgebra::allocator::Allocator<Z>
        + nalgebra::allocator::Allocator<SP>
        + nalgebra::allocator::Allocator<N, SP>
        + nalgebra::allocator::Allocator<Z, SP>
        + nalgebra::allocator::Allocator<N, N>
        + nalgebra::allocator::Allocator<Z, Z>
        + nalgebra::allocator::Allocator<N, Z>
        + nalgebra::allocator::Allocator<SP, SP>
        + nalgebra::allocator::Allocator<Const<1>, Z>
        + nalgebra::allocator::Allocator<Z, N>
        + nalgebra::allocator::Allocator<Const<1>, N>
        + nalgebra::allocator::Allocator<SP, N>
        + nalgebra::allocator::Allocator<SP, Z>,
{
    pub fn new(weights: UTWeights<SP, T>, regularization_factor: T, n_dim: N, z_dim: Z) -> Self {
        let sp_dim = SP::name();

        Self {
            weights,
            regularization_factor,
            sigma_points: OMatrix::zeros_generic(n_dim, sp_dim),
            state_deviations: OMatrix::zeros_generic(n_dim, sp_dim),
            measurement_deviations: OMatrix::zeros_generic(z_dim, sp_dim),
            pxz_buffer: OMatrix::zeros_generic(n_dim, z_dim),
            z_pred_buffer: OVector::zeros_generic(z_dim, Const::<1>),
            x_buffer: OVector::zeros_generic(n_dim, Const::<1>),
            transformed: OMatrix::zeros_generic(n_dim, sp_dim),
            z_sigma_buffer: OMatrix::zeros_generic(z_dim, sp_dim),
        }
    }

    pub fn predict_covariance(
        &mut self,
        process_noise_cov: &OMatrix<T, N, N>,
    ) -> Result<Cholesky<T, N>, UkfError> {
        let n_sigmas = SP::dim();

        for i in 0..n_sigmas {
            self.state_deviations
                .column_mut(i)
                .copy_from(&(self.transformed.column(i) - &self.x_buffer));
        }

        let mut p_cov = process_noise_cov.clone_owned();

        for i in 0..n_sigmas {
            let weight = self.weights.w_covar[i];
            let deviation = self.state_deviations.column(i);
            p_cov.gemm(weight, &deviation, &deviation.transpose(), T::one());
        }

        match Cholesky::new(p_cov.clone()) {
            Some(ch) => Ok(ch),
            None => {
                let regularized =
                    p_cov + OMatrix::<T, N, N>::identity() * self.regularization_factor;
                Cholesky::new(regularized).ok_or(UkfError::CholeskyDecompositionFailed)
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn update<WM>(
        &mut self,
        state_sqrt_p: &Cholesky<T, N>,
        measurement_noise_cov: &OMatrix<T, Z, Z>,
        weighted_mean_calculator: &WM,
    ) -> Result<(OMatrix<T, N, Z>, OVector<T, Z>, Cholesky<T, N>), UkfError>
    where
        WM: WeightedMean<Z, T>,
    {
        let n_sigmas = SP::dim();

        weighted_mean_calculator.weighted_mean(
            &self.z_sigma_buffer,
            &self.weights.w_mean,
            &mut self.z_pred_buffer,
        );

        let mut pzz_cov = measurement_noise_cov.clone_owned();
        for i in 0..n_sigmas {
            let weight = self.weights.w_covar[i];
            let deviation = self.measurement_deviations.column(i);
            pzz_cov.gemm(weight, &deviation, &deviation.transpose(), T::one());
        }

        let pzz_cholesky = match Cholesky::new(pzz_cov.clone()) {
            Some(ch) => ch,
            None => {
                let regularized =
                    pzz_cov + OMatrix::<T, Z, Z>::identity() * self.regularization_factor;
                Cholesky::new(regularized).ok_or(UkfError::CholeskyDecompositionFailed)?
            }
        };

        self.pxz_buffer.fill(T::zero());
        for i in 0..n_sigmas {
            self.pxz_buffer.gemm(
                self.weights.w_covar[i],
                &self.state_deviations.column(i),
                &self.measurement_deviations.column(i).transpose(),
                T::one(),
            );
        }

        let mut pxz_t = self.pxz_buffer.transpose();
        pzz_cholesky.solve_mut(&mut pxz_t);
        let k_gain = pxz_t.transpose();

        let mut new_sqrt_p = state_sqrt_p.clone();
        let l_z = pzz_cholesky.l();
        let s_matrix = &k_gain * l_z;

        for i in 0..Z::dim() {
            let s_col = s_matrix.column(i);
            new_sqrt_p.rank_one_update(&s_col, -T::one());
        }

        Ok((k_gain, self.z_pred_buffer.clone(), new_sqrt_p))
    }
}
