use nalgebra::{Cholesky, Const, OMatrix, OVector, RealField};

use crate::kf::averaging::WeightedMean;
use crate::kf::error::UkfError;
use crate::sr_ukf::UTWeights;

/// The core UKF engine, encapsulating the mathematical steps of the Unscented Kalman Filter.
///
/// This engine is stateless regarding the filter's actual state (e.g., position, attitude)
/// but contains the buffers and logic required to perform core UKF calculations on raw vectors.
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

    /// Regularization factor for the covariance update.
    pub(crate) regularization_factor: T,

    /// Buffer for sigma points.
    pub(crate) sigma_points: OMatrix<T, N, SP>,

    /// Buffer for state deviations.
    pub(crate) state_deviations: OMatrix<T, N, SP>,

    /// Buffer for measurement deviations.
    pub(crate) measurement_deviations: OMatrix<T, Z, SP>,

    /// Buffer for cross-covariance (Pxz).
    pub(crate) pxz_buffer: OMatrix<T, N, Z>,

    /// Buffer for predicted measurements mean (z_pred).
    pub(crate) z_pred_buffer: OVector<T, Z>,

    /// Scratch buffer for weighted mean calculation.
    pub(crate) w_mean_scratch: OVector<T, SP>,

    /// Scratch buffer for weighted covariance calculation.
    pub(crate) w_covar_scratch: OVector<T, SP>,

    /// Buffer for state vector (x_buffer).
    pub(crate) x_buffer: OVector<T, N>,

    /// Buffer for transformed sigma points (after process model).
    pub(crate) transformed: OMatrix<T, N, SP>,

    /// Buffer for measurement sigma points (after measurement model).
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
    ///
    /// # Arguments
    /// * `weights` - The sigma point weights.
    /// * `regularization_factor` - The regularization factor for the covariance update.
    /// * `n_dim` - The dimension of the state vector.
    /// * `z_dim` - The dimension of the measurement vector.
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
            w_mean_scratch: OVector::zeros_generic(sp_dim, Const::<1>),
            w_covar_scratch: OVector::zeros_generic(sp_dim, Const::<1>),
            x_buffer: OVector::zeros_generic(n_dim, Const::<1>),
            transformed: OMatrix::zeros_generic(n_dim, sp_dim),
            z_sigma_buffer: OMatrix::zeros_generic(z_dim, sp_dim),
        }
    }

    /// Predicts the new state covariance given transformed sigma points, their predicted mean,
    /// and the process noise covariance.
    ///
    /// # Arguments
    /// * `process_noise_cov` - The full process noise covariance matrix (Q).
    ///
    /// # Returns
    /// A `Cholesky` decomposition of the predicted state covariance.
    pub fn predict_covariance(
        &mut self,
        process_noise_cov: &OMatrix<T, N, N>,
    ) -> Result<Cholesky<T, N>, UkfError> {
        let n_sigmas = SP::dim();

        // Calculate state deviations
        for i in 0..n_sigmas {
            self.state_deviations
                .column_mut(i)
                .copy_from(&(self.transformed.column(i) - &self.x_buffer));
        }

        // Initialize P_k|k-1 with process noise
        let mut p_cov = process_noise_cov.to_owned();

        // Sum weighted outer products of state deviations
        for i in 0..n_sigmas {
            let weight = self.weights.w_covar[i];
            let deviation = self.state_deviations.column(i);
            p_cov.gemm(weight, &deviation, &deviation.transpose(), T::one());
        }

        match Cholesky::new(p_cov.clone()) {
            Some(ch) => Ok(ch),
            None => {
                p_cov += OMatrix::<T, N, N>::identity() * self.regularization_factor;
                Cholesky::new(p_cov).ok_or(UkfError::CholeskyDecompositionFailed)
            }
        }
    }

    /// Performs the UKF update step, calculating the Kalman gain, state correction, and updated covariance.
    ///
    /// # Arguments
    /// * `state_sqrt_p` - The Cholesky decomposition of the current state covariance.
    /// * `state_deviations` - The matrix of state deviations.
    /// * `measurement_noise_cov` - The full measurement noise covariance matrix (R).
    /// * `weighted_mean_calculator` - A `WeightedMean` implementation for calculating the mean of measurement sigmas.
    ///
    /// # Returns
    /// A tuple containing:
    /// (Kalman Gain, Predicted measurement mean, Cholesky decomposition of the updated state covariance).
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

        // 1. Calculate predicted measurement mean (z_pred)
        weighted_mean_calculator.weighted_mean(
            &self.z_sigma_buffer,
            &self.weights.w_mean,
            &mut self.z_pred_buffer,
        );

        // 2. Measurement deviations are now pre-calculated by the caller (e.g., MultiplicativeUKF)
        //    using the appropriate manifold-aware residual function.

        // 3. Calculate innovation covariance (Pzz)
        let mut pzz_cov = measurement_noise_cov.to_owned();
        for i in 0..n_sigmas {
            let weight = self.weights.w_covar[i];
            let deviation = self.measurement_deviations.column(i);
            pzz_cov.gemm(weight, &deviation, &deviation.transpose(), T::one());
        }

        let pzz_cholesky = match Cholesky::new(pzz_cov.clone()) {
            Some(ch) => ch,
            None => {
                pzz_cov += OMatrix::<T, Z, Z>::identity() * self.regularization_factor;
                Cholesky::new(pzz_cov).ok_or(UkfError::CholeskyDecompositionFailed)?
            }
        };

        // 4. Calculate cross-covariance (Pxz)
        self.pxz_buffer.fill(T::zero());
        for i in 0..n_sigmas {
            self.pxz_buffer.gemm(
                self.weights.w_covar[i],
                &self.state_deviations.column(i),
                &self.measurement_deviations.column(i).transpose(),
                T::one(),
            );
        }

        // 5. Calculate Kalman Gain (K = Pxz * Pzz^-1)
        let mut pxz_t = self.pxz_buffer.transpose();
        pzz_cholesky.solve_mut(&mut pxz_t);
        let k_gain = pxz_t.transpose();

        // 6. Update state covariance using Cholesky rank-one downdates
        let mut new_sqrt_p = state_sqrt_p.clone();

        // Calculate S = K * L_z, where L_z is the Cholesky factor of Pzz
        let l_z = pzz_cholesky.l();
        let s_matrix = &k_gain * l_z;

        // Perform rank-one downdates for each column of S
        for i in 0..Z::dim() {
            let s_col = s_matrix.column(i);
            // Downdate with sigma = -1.0
            new_sqrt_p.rank_one_update(&s_col, -T::one());
        }

        Ok((k_gain, self.z_pred_buffer.clone(), new_sqrt_p))
    }
}
