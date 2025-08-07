use core::marker::PhantomData;
use nalgebra::{
    allocator::Allocator, base::OMatrix, base::OVector, Cholesky, Const, DefaultAllocator, DimAdd,
    DimMin, DimMinimum, DimName, RealField,
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
        + Allocator<Z, Z>
        + Allocator<Z, SP::SigmaCount>
        + Allocator<N, SP::SigmaCount>
        + Allocator<SP::SigmaCount>
        + Allocator<SP::SigmaCount, N>
        + Allocator<N, Z>,
{
    state: S,
    sqrt_p: Cholesky<T, N>,
    process_model: P,
    process_noise_sqrt: OMatrix<T, N, N>,
    measurement_model: M,
    measurement_noise_sqrt: OMatrix<T, Z, Z>,
    sigma_point_generator: SP,
    weights: UTWeights<SP::SigmaCount, T>,
    w_mean_scratch: OVector<T, SP::SigmaCount>,
    w_covar_scratch: OVector<T, SP::SigmaCount>,
    sigma_points: OMatrix<T, N, SP::SigmaCount>,
    transformed: OMatrix<T, N, SP::SigmaCount>,
    x_buffer: OVector<T, N>,

    // Additional buffer fields to reduce memory allocations
    z_sigma_buffer: OMatrix<T, Z, SP::SigmaCount>,
    z_pred_buffer: OVector<T, Z>,
    pzz_buffer: OMatrix<T, Z, Z>,
    pzz_sqrt_buffer: OMatrix<T, Z, Z>,
    pxz_buffer: OMatrix<T, N, Z>,
    k_gain_buffer: OMatrix<T, N, Z>,
    deviation_buffer: OVector<T, N>,
    innovation_buffer: OVector<T, Z>,
    dz_buffer: OVector<T, Z>,

    regularization_factor: T,
    _phantom_c: PhantomData<C>,
}

impl<S, N, P, C, M, Z, SP, T> SquareRootUKF<S, N, P, C, M, Z, SP, T>
where
    S: State<N, T> + Clone,
    N: DimName + DimMin<N, Output = N> + DimMin<<SP::SigmaCount as DimAdd<N>>::Output>,
    P: ProcessModel<N, C, T>,
    C: DimName,
    M: MeasurementModel<N, Z, T>,
    Z: DimName,
    SP: SigmaPointsInPlace<N, T>,
    SP::SigmaCount:
        DimName + DimMin<N> + DimMin<SP::SigmaCount, Output = SP::SigmaCount> + DimAdd<N>,
    <SP::SigmaCount as DimAdd<N>>::Output: DimName,
    T: RealField + Copy,
    DefaultAllocator: Allocator<N>
        + Allocator<N, N>
        + Allocator<C>
        + Allocator<Z>
        + Allocator<Z, Z>
        + Allocator<Z, SP::SigmaCount>
        + Allocator<N, SP::SigmaCount>
        + Allocator<SP::SigmaCount>
        + Allocator<SP::SigmaCount, N>
        + Allocator<N, Z>
        + Allocator<Z, N>
        + Allocator<Const<1>, Z>
        + Allocator<Const<1>, N>
        + Allocator<N, SP::SigmaCount>
        + Allocator<Z, SP::SigmaCount>
        + Allocator<SP::SigmaCount, SP::SigmaCount>
        + Allocator<DimMinimum<SP::SigmaCount, N>, N>
        + Allocator<DimMinimum<SP::SigmaCount, N>, SP::SigmaCount>
        + Allocator<DimMinimum<N, N>, N>
        + Allocator<DimMinimum<SP::SigmaCount, N>>
        + Allocator<DimMinimum<N, N>>
        + Allocator<N, <SP::SigmaCount as DimAdd<N>>::Output>
        + Allocator<DimMinimum<N, <SP::SigmaCount as DimAdd<N>>::Output>>
        + Allocator<
            DimMinimum<N, <SP::SigmaCount as DimAdd<N>>::Output>,
            <SP::SigmaCount as DimAdd<N>>::Output,
        > + Allocator<<SP::SigmaCount as DimAdd<N>>::Output, N>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_state: S,
        initial_covariance_sqrt: OMatrix<T, N, N>,
        process_model: P,
        process_noise_sqrt: OMatrix<T, N, N>,
        measurement_model: M,
        measurement_noise_sqrt: OMatrix<T, Z, Z>,
        sigma_point_generator: SP,
        weights: UTWeights<SP::SigmaCount, T>,
    ) -> Self {
        let w_mean_scratch = weights.w_mean.clone();
        let w_covar_scratch = weights.w_covar.clone();
        let regularization_factor = T::default_epsilon() * T::from_f64(100.0).unwrap_or(T::one());

        // Initialize state vector buffer
        let mut x_buffer = OVector::<T, N>::zeros();
        initial_state.write_into(&mut x_buffer);

        Self {
            state: initial_state,
            sqrt_p: Cholesky::new(initial_covariance_sqrt)
                .expect("Initial covariance must be positive-definite"),
            process_model,
            process_noise_sqrt,
            measurement_model,
            measurement_noise_sqrt,
            sigma_point_generator,
            weights,
            w_mean_scratch,
            w_covar_scratch,
            sigma_points: OMatrix::<T, N, SP::SigmaCount>::zeros(),
            transformed: OMatrix::<T, N, SP::SigmaCount>::zeros(),
            x_buffer,

            // Initialize additional buffer fields to avoid allocations
            z_sigma_buffer: OMatrix::<T, Z, SP::SigmaCount>::zeros(),
            z_pred_buffer: OVector::<T, Z>::zeros(),
            pzz_buffer: OMatrix::<T, Z, Z>::zeros(),
            pzz_sqrt_buffer: OMatrix::<T, Z, Z>::zeros(),
            pxz_buffer: OMatrix::<T, N, Z>::zeros(),
            k_gain_buffer: OMatrix::<T, N, Z>::zeros(),
            deviation_buffer: OVector::<T, N>::zeros(),
            innovation_buffer: OVector::<T, Z>::zeros(),
            dz_buffer: OVector::<T, Z>::zeros(),

            regularization_factor,
            _phantom_c: PhantomData,
        }
    }

    pub fn with_regularization_factor(mut self, factor: T) -> Self {
        self.regularization_factor = factor;
        self
    }

    fn reset_scratch_weights(&mut self) {
        self.w_mean_scratch.copy_from(&self.weights.w_mean);
        self.w_covar_scratch.copy_from(&self.weights.w_covar);
    }

    pub fn predict(&mut self, dt: T, control: Option<&OVector<T, C>>) {
        self.reset_scratch_weights();
        self.state.write_into(&mut self.x_buffer);
        self.generate_sigma_points();
        self.predict_mean(dt, control);
        self.predict_covariance();
    }

    pub fn get_state(&self) -> &S {
        &self.state
    }

    pub fn get_sqrt_covariance(&self) -> OMatrix<T, N, N> {
        self.sqrt_p.l()
    }

    pub fn update(&mut self, measurement: &OVector<T, Z>) {
        self.reset_scratch_weights();
        self.generate_sigma_points();
        let n_sigmas = self.sigma_points.ncols();

        // Transform sigma points to measurement space
        for i in 0..n_sigmas {
            let xi = self.sigma_points.column(i).clone_owned();
            let mut zi_column = self.z_sigma_buffer.column_mut(i);
            let zi = self.measurement_model.measure(&xi);
            zi_column.copy_from(&zi);
        }

        // Calculate predicted measurement and cross-covariance
        self.z_pred_buffer.fill(T::zero());
        self.pxz_buffer.fill(T::zero());

        for i in 0..n_sigmas {
            let weight_mean = self.w_mean_scratch[i];
            let zi = self.z_sigma_buffer.column(i);
            self.z_pred_buffer.axpy(weight_mean, &zi, T::one());
        }

        for i in 0..n_sigmas {
            let weight_covar = self.w_covar_scratch[i];
            self.deviation_buffer
                .copy_from(&self.sigma_points.column(i));
            self.deviation_buffer -= &self.x_buffer;
            self.dz_buffer.copy_from(&self.z_sigma_buffer.column(i));
            self.dz_buffer -= &self.z_pred_buffer;
            self.pxz_buffer.gemm(
                weight_covar,
                &self.deviation_buffer,
                &self.dz_buffer.transpose(),
                T::one(),
            );
        }

        // Calculate innovation covariance square root
        let mut pzz_chol = Cholesky::pack_dirty(self.measurement_noise_sqrt.clone());
        for i in 0..n_sigmas {
            self.dz_buffer.copy_from(&self.z_sigma_buffer.column(i));
            self.dz_buffer -= &self.z_pred_buffer;
            let weight = self.w_covar_scratch[i];
            if weight.abs() > T::default_epsilon() {
                pzz_chol.rank_one_update(&self.dz_buffer, weight);
            }
        }
        self.pzz_sqrt_buffer.copy_from(pzz_chol.l_dirty());

        // Compute full innovation covariance
        self.pzz_sqrt_buffer
            .mul_to(&self.pzz_sqrt_buffer.transpose(), &mut self.pzz_buffer);

        // Add regularization and compute inverse
        let pzz_inv = self.pzz_buffer.clone().try_inverse().unwrap_or_else(|| {
            let mut reg = self.pzz_buffer.clone();
            for i in 0..Z::dim() {
                reg[(i, i)] += self.regularization_factor;
            }
            reg.try_inverse()
                .expect("Matrix should be invertible after regularization")
        });

        // Calculate Kalman gain
        self.k_gain_buffer
            .copy_from(&(self.pxz_buffer.clone() * pzz_inv));

        // Update state
        self.innovation_buffer.copy_from(
            &self
                .measurement_model
                .residual(&self.z_pred_buffer, measurement),
        );
        self.x_buffer.gemm(
            T::one(),
            &self.k_gain_buffer,
            &self.innovation_buffer,
            T::one(),
        );
        self.state = S::from_vector(self.x_buffer.clone());

        // Update state covariance
        let mut p_chol = self.sqrt_p.clone();
        let u = self.k_gain_buffer.clone() * self.pzz_sqrt_buffer.clone();
        for j in 0..Z::dim() {
            let u_col = u.column(j);
            p_chol.rank_one_update(&u_col, -T::one());
        }

        let mut p_sqrt = p_chol.l_dirty().clone_owned();
        let n_dim = N::dim();
        for i in 0..n_dim {
            p_sqrt[(i, i)] += self.regularization_factor;
        }
        self.sqrt_p = Cholesky::pack_dirty(p_sqrt);
    }

    fn generate_sigma_points(&mut self) {
        self.sigma_point_generator.generate_into(
            &self.x_buffer,
            &self.sqrt_p,
            &mut self.sigma_points,
            &mut self.w_mean_scratch,
            &mut self.w_covar_scratch,
        );
    }

    fn predict_mean(&mut self, dt: T, control: Option<&OVector<T, C>>) {
        let n_sigmas = self.sigma_points.ncols();

        // First iterate through sigma points and transform them
        for i in 0..n_sigmas {
            // Need to clone the column view into an owned vector for the predict function
            let xi = self.sigma_points.column(i).clone_owned();

            // Get a mutable column to the transformed sigma points
            let mut yi_column = self.transformed.column_mut(i);

            // Predict directly into the column without additional allocations
            let yi = self.process_model.predict(&xi, dt, control);
            yi_column.copy_from(&yi);
        }

        // Zero out the state prediction buffer
        self.x_buffer.fill(T::zero());

        // Accumulate weighted sigma points directly into x_buffer
        for i in 0..n_sigmas {
            // Use references instead of cloning
            let weight = self.w_mean_scratch[i];
            let sigma_i = self.transformed.column(i);

            // Accumulate weighted contribution
            self.x_buffer.axpy(weight, &sigma_i, T::one());
        }

        // Set the state from the buffer
        self.state = S::from_vector(self.x_buffer.clone());
    }

    fn predict_covariance(&mut self) {
        // Start with the process noise as the initial covariance
        let mut chol = Cholesky::pack_dirty(self.process_noise_sqrt.clone());

        // Apply rank-1 updates for each sigma point
        for i in 0..self.sigma_points.ncols() {
            self.deviation_buffer.copy_from(&self.transformed.column(i));
            self.deviation_buffer -= &self.x_buffer;

            let weight = self.w_covar_scratch[i];
            if weight.abs() > T::default_epsilon() {
                chol.rank_one_update(&self.deviation_buffer, weight);
            }
        }

        // Ensure positive-definiteness by adding regularization
        let mut l_sqrt = chol.l_dirty().clone_owned();
        let n_dim = N::dim();
        for i in 0..n_dim {
            l_sqrt[(i, i)] += self.regularization_factor;
        }

        self.sqrt_p = Cholesky::pack_dirty(l_sqrt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sigma_points::MerweScaled;
    use approx::assert_relative_eq;
    use nalgebra::Matrix;
    use nalgebra::{Const, OMatrix, Vector1, Vector2, U1, U2};

    #[derive(Clone, Debug)]
    struct PVState(Vector2<f64>);

    impl State<U2, f64> for PVState {
        fn into_vector(self) -> Vector2<f64> {
            self.0
        }

        fn write_into(&self, buf: &mut Vector2<f64>) {
            buf.copy_from(&self.0);
        }

        fn from_vector(v: Vector2<f64>) -> Self {
            PVState(v)
        }
    }

    struct LinearProcessModel;

    impl ProcessModel<U2, U1, f64> for LinearProcessModel {
        fn predict<S: nalgebra::storage::Storage<f64, U2, Const<1>>>(
            &self,
            state: &Matrix<f64, U2, Const<1>, S>,
            dt: f64,
            _control: Option<&Vector1<f64>>,
        ) -> Vector2<f64> {
            let f = OMatrix::<f64, U2, U2>::new(1.0, dt, 0.0, 1.0);
            f * state.clone_owned()
        }
    }

    struct LinearMeasurementModel;

    impl MeasurementModel<U2, U1, f64> for LinearMeasurementModel {
        fn measure(&self, x: &OVector<f64, U2>) -> OVector<f64, U1> {
            OVector::<f64, U1>::new(x[0])
        }

        fn residual(
            &self,
            z_pred: &OVector<f64, U1>,
            z_meas: &OVector<f64, U1>,
        ) -> OVector<f64, U1> {
            z_meas - z_pred
        }
    }

    #[test]
    fn test_construction() {
        let initial_state = PVState(Vector2::new(1.0, 2.0));
        let chol_p = OMatrix::<f64, U2, U2>::identity();
        let process_noise = OMatrix::<f64, U2, U2>::identity() * 0.1;
        let measurement_noise = OMatrix::<f64, U1, U1>::identity() * 0.1;

        let weights = UTWeights::<Const<5>, f64>::new(U2::dim(), 0.001, 2.0, 0.0);

        let _ukf = SquareRootUKF::new(
            initial_state.clone(),
            chol_p.clone(),
            LinearProcessModel,
            process_noise.clone(),
            LinearMeasurementModel,
            measurement_noise.clone(),
            MerweScaled {
                alpha: 0.001,
                beta: 2.0,
                kappa: 0.0,
            },
            weights.clone(),
        );

        let ukf_with_config = SquareRootUKF::new(
            initial_state,
            chol_p,
            LinearProcessModel,
            process_noise.clone(),
            LinearMeasurementModel,
            measurement_noise,
            MerweScaled {
                alpha: 0.001,
                beta: 2.0,
                kappa: 0.0,
            },
            weights,
        )
        .with_regularization_factor(0.5);

        assert_eq!(ukf_with_config.regularization_factor, 0.5);
    }

    #[test]
    fn test_predict_mean_updates_state() {
        let initial_state = PVState(Vector2::new(1.0, 2.0));
        let chol_p = OMatrix::<f64, U2, U2>::identity();
        let process_noise_sqrt = OMatrix::<f64, U2, U2>::identity() * 0.1;
        let measurement_noise_sqrt = OMatrix::<f64, U1, U1>::identity() * 0.1;

        let sigma_generator = MerweScaled {
            alpha: 0.001,
            beta: 2.0,
            kappa: 0.0,
        };
        let weights = UTWeights::<Const<5>, f64>::new(U2::dim(), 0.001, 2.0, 0.0);

        let mut ukf = SquareRootUKF::new(
            initial_state,
            chol_p,
            LinearProcessModel,
            process_noise_sqrt.clone(),
            LinearMeasurementModel,
            measurement_noise_sqrt,
            sigma_generator,
            weights,
        );

        ukf.predict(0.1, None);

        assert_relative_eq!(ukf.state.0, Vector2::new(1.2, 2.0), epsilon = 1e-9);
        for i in 0..2 {
            for j in (i + 1)..2 {
                assert_relative_eq!(ukf.get_sqrt_covariance()[(i, j)], 0.0, epsilon = 1e-12);
            }
        }
        assert!(ukf.get_sqrt_covariance().norm() > 0.0);
    }

    #[test]
    fn test_predict_with_linear_model() {
        let initial_state = PVState(Vector2::new(1.0, 2.0));
        let chol_p = OMatrix::<f64, U2, U2>::identity();
        let process_noise_sqrt = OMatrix::<f64, U2, U2>::identity() * 0.1;
        let measurement_noise_sqrt = OMatrix::<f64, U1, U1>::identity() * 0.1;

        let dt = 0.5;
        let expected_state = Vector2::new(2.0, 2.0);

        let mut ukf = SquareRootUKF::new(
            initial_state,
            chol_p,
            LinearProcessModel,
            process_noise_sqrt.clone(),
            LinearMeasurementModel,
            measurement_noise_sqrt,
            MerweScaled {
                alpha: 0.001,
                beta: 2.0,
                kappa: 0.0,
            },
            UTWeights::<Const<5>, f64>::new(U2::dim(), 0.001, 2.0, 0.0),
        );

        ukf.predict(dt, None);
        assert_relative_eq!(ukf.state.0, expected_state, epsilon = 1e-9);
    }
}
