use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use nalgebra::allocator::Allocator;
use nalgebra::{Cholesky, Const, DefaultAllocator, DimName, OMatrix, OVector, RealField};

use crate::manifold::{InitialGuess, Manifold, ManifoldMeasurement, ManifoldProcess, MeanError};
use crate::sigma_points::SigmaPointsInPlace;
use crate::sigma_points::UTWeights;

use super::averaging::{LinearAveraging, WeightedMean};
use super::engine::UKFEngine;
use super::error::UkfError;

/// An Unscented Kalman Filter operating on manifold-valued states.
///
/// The filter keeps a nominal state on the manifold and an error covariance in the tangent space.
/// Sigma points are generated in the tangent space, mapped onto the manifold via `retract`, and
/// then passed through the process and measurement models.
#[derive(Clone, Debug)]
pub struct UnscentedKalmanFilter<
    Nominal,
    Process,
    Measurement,
    TangentDim,
    ControlDim,
    MeasDim,
    SigmaGen,
    T,
> where
    TangentDim: DimName,
    ControlDim: DimName,
    MeasDim: DimName,
    Nominal: Manifold<TangentDim, T> + Clone,
    Process: ManifoldProcess<Nominal, ControlDim, T>,
    Measurement: ManifoldMeasurement<Nominal, TangentDim, MeasDim, T>,
    SigmaGen: SigmaPointsInPlace<TangentDim, T>,
    T: RealField + Copy,
    DefaultAllocator: Allocator<TangentDim>
        + Allocator<TangentDim, TangentDim>
        + Allocator<ControlDim>
        + Allocator<MeasDim>
        + Allocator<MeasDim, MeasDim>
        + Allocator<MeasDim, SigmaGen::SigmaCount>
        + Allocator<TangentDim, SigmaGen::SigmaCount>
        + Allocator<SigmaGen::SigmaCount>
        + Allocator<TangentDim, MeasDim>
        + Allocator<MeasDim, TangentDim>
        + Allocator<Const<1>, TangentDim>
        + Allocator<Const<1>, MeasDim>
        + Allocator<SigmaGen::SigmaCount, SigmaGen::SigmaCount>
        + Allocator<SigmaGen::SigmaCount, TangentDim>
        + Allocator<SigmaGen::SigmaCount, MeasDim>,
{
    nominal_state: Nominal,
    error_covariance_sqrt: Cholesky<T, TangentDim>,
    process_model: Process,
    process_noise_cov: OMatrix<T, TangentDim, TangentDim>,
    measurement_model: Measurement,
    measurement_noise_cov: OMatrix<T, MeasDim, MeasDim>,
    sigma_generator: SigmaGen,
    engine: UKFEngine<TangentDim, MeasDim, SigmaGen::SigmaCount, T>,
    nominal_sigmas: Vec<Nominal>,
    predicted_sigmas: Vec<Nominal>,
    weighted_mean: LinearAveraging,
    _phantom_control: PhantomData<ControlDim>,
}

impl<Nominal, Process, Measurement, TangentDim, ControlDim, MeasDim, SigmaGen, T>
    UnscentedKalmanFilter<
        Nominal,
        Process,
        Measurement,
        TangentDim,
        ControlDim,
        MeasDim,
        SigmaGen,
        T,
    >
where
    TangentDim: DimName,
    ControlDim: DimName,
    MeasDim: DimName,
    Nominal: Manifold<TangentDim, T> + Clone,
    Process: ManifoldProcess<Nominal, ControlDim, T>,
    Measurement: ManifoldMeasurement<Nominal, TangentDim, MeasDim, T>,
    SigmaGen: SigmaPointsInPlace<TangentDim, T>,
    T: RealField + Copy,
    DefaultAllocator: Allocator<TangentDim>
        + Allocator<TangentDim, TangentDim>
        + Allocator<ControlDim>
        + Allocator<MeasDim>
        + Allocator<MeasDim, MeasDim>
        + Allocator<MeasDim, SigmaGen::SigmaCount>
        + Allocator<TangentDim, SigmaGen::SigmaCount>
        + Allocator<SigmaGen::SigmaCount>
        + Allocator<TangentDim, MeasDim>
        + Allocator<MeasDim, TangentDim>
        + Allocator<Const<1>, TangentDim>
        + Allocator<Const<1>, MeasDim>
        + Allocator<SigmaGen::SigmaCount, SigmaGen::SigmaCount>
        + Allocator<SigmaGen::SigmaCount, TangentDim>
        + Allocator<SigmaGen::SigmaCount, MeasDim>,
{
    /// Construct a new UKF instance.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_nominal: Nominal,
        initial_error_covariance_sqrt: OMatrix<T, TangentDim, TangentDim>,
        process_model: Process,
        process_noise_cov: OMatrix<T, TangentDim, TangentDim>,
        measurement_model: Measurement,
        measurement_noise_cov: OMatrix<T, MeasDim, MeasDim>,
        sigma_generator: SigmaGen,
        weights: UTWeights<SigmaGen::SigmaCount, T>,
    ) -> Self {
        let n_sigmas = SigmaGen::SigmaCount::dim();
        let regularization_factor = T::from_subset(&1e-6);

        let engine = UKFEngine::new(
            weights,
            regularization_factor,
            TangentDim::name(),
            MeasDim::name(),
        );

        Self {
            nominal_state: initial_nominal.clone(),
            error_covariance_sqrt: Cholesky::new(initial_error_covariance_sqrt)
                .expect("initial error covariance must be positive definite"),
            process_model,
            process_noise_cov,
            measurement_model,
            measurement_noise_cov,
            sigma_generator,
            engine,
            nominal_sigmas: vec![initial_nominal.clone(); n_sigmas],
            predicted_sigmas: vec![initial_nominal; n_sigmas],
            weighted_mean: LinearAveraging,
            _phantom_control: PhantomData,
        }
    }

    /// Override the regularization factor used when Cholesky fails.
    pub fn with_regularization_factor(mut self, factor: T) -> Self {
        self.engine.regularization_factor = factor;
        self
    }

    /// Return the current nominal state.
    pub fn nominal_state(&self) -> &Nominal {
        &self.nominal_state
    }

    /// Return the current error covariance (full matrix form).
    pub fn error_covariance(&self) -> OMatrix<T, TangentDim, TangentDim> {
        let l = self.error_covariance_sqrt.l();
        &l * l.transpose()
    }

    /// Predict the state forward.
    pub fn predict(
        &mut self,
        dt: T,
        control: Option<&OVector<T, ControlDim>>,
    ) -> Result<(), UkfError> {
        let engine = &mut self.engine;

        self.sigma_generator.generate_into(
            &OVector::zeros_generic(TangentDim::name(), Const::<1>),
            &self.error_covariance_sqrt,
            &mut engine.sigma_points,
            &mut engine.weights.w_mean,
            &mut engine.weights.w_covar,
        );

        for (i, nominal_sigma) in self.nominal_sigmas.iter_mut().enumerate() {
            let tangent_sigma = engine.sigma_points.column(i);
            *nominal_sigma = self.nominal_state.retract(&tangent_sigma.into_owned());
        }

        for (i, predicted) in self.predicted_sigmas.iter_mut().enumerate() {
            *predicted = self
                .process_model
                .predict(&self.nominal_sigmas[i], dt, control);
        }

        let weighted_mean_result = Nominal::weighted_mean(
            &self.predicted_sigmas,
            engine.weights.w_mean.as_slice(),
            T::from_subset(&1e-9),
            InitialGuess::MaxWeight,
            100,
        );

        self.nominal_state = match weighted_mean_result {
            Ok(mean) => mean,
            Err(MeanError::NotConverged) => self.predicted_sigmas[0].clone(),
            Err(
                MeanError::EmptyInput
                | MeanError::NoPositiveWeights
                | MeanError::InvalidTolerance
                | MeanError::LengthMismatch
                | MeanError::IndexOutOfBounds,
            ) => return Err(UkfError::MeanComputationFailed),
        };

        Nominal::batch_local_into_matrix(
            &self.nominal_state,
            &self.predicted_sigmas,
            &mut engine.transformed,
        );

        self.weighted_mean.weighted_mean(
            &engine.transformed,
            &engine.weights.w_mean,
            &mut engine.x_buffer,
        );

        self.error_covariance_sqrt = engine.predict_covariance(&self.process_noise_cov)?;
        Ok(())
    }

    /// Update with a measurement.
    pub fn update(&mut self, measurement: &OVector<T, MeasDim>) -> Result<(), UkfError> {
        let engine = &mut self.engine;

        self.sigma_generator.generate_into(
            &OVector::zeros_generic(TangentDim::name(), Const::<1>),
            &self.error_covariance_sqrt,
            &mut engine.sigma_points,
            &mut engine.weights.w_mean,
            &mut engine.weights.w_covar,
        );

        for (i, nominal_sigma) in self.nominal_sigmas.iter_mut().enumerate() {
            let tangent_sigma = engine.sigma_points.column(i);
            *nominal_sigma = self.nominal_state.retract(&tangent_sigma.into_owned());
        }

        for i in 0..self.nominal_sigmas.len() {
            let z_sigma = self.measurement_model.measure(&self.nominal_sigmas[i]);
            engine.z_sigma_buffer.column_mut(i).copy_from(&z_sigma);
        }

        let mut z_pred = OVector::<T, MeasDim>::zeros();
        self.weighted_mean.weighted_mean(
            &engine.z_sigma_buffer,
            &engine.weights.w_mean,
            &mut z_pred,
        );

        let innovation = self.measurement_model.innovation(measurement, &z_pred);

        for i in 0..self.nominal_sigmas.len() {
            let z_sigma = engine.z_sigma_buffer.column(i);
            let residual = self
                .measurement_model
                .residual(&z_pred, &z_sigma.into_owned());
            engine
                .measurement_deviations
                .column_mut(i)
                .copy_from(&residual);
        }

        engine.state_deviations.copy_from(&engine.sigma_points);

        let (kalman_gain, _z_pred, new_error_covariance_sqrt) = engine.update(
            &self.error_covariance_sqrt,
            &self.measurement_noise_cov,
            &self.weighted_mean,
        )?;

        let error_update = &kalman_gain * innovation;

        self.nominal_state = self.nominal_state.retract(&error_update);
        self.error_covariance_sqrt = new_error_covariance_sqrt;

        Ok(())
    }

    /// Access the current state and covariance.
    pub fn state_with_covariance(&self) -> (Nominal, OMatrix<T, TangentDim, TangentDim>) {
        (self.nominal_state.clone(), self.error_covariance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::euclidean::EuclideanManifold;
    use crate::manifold::{ManifoldMeasurement, ManifoldProcess};
    use crate::sigma_points::MerweScaledSigmaPoints;
    use nalgebra::{Matrix1, Matrix2, Vector1, Vector2, U1, U2};

    #[derive(Clone)]
    struct ConstantVelocityProcess {
        transition: Matrix2<f64>,
    }

    impl ConstantVelocityProcess {
        fn new(dt: f64) -> Self {
            Self {
                transition: Matrix2::new(1.0, dt, 0.0, 1.0),
            }
        }
    }

    impl ManifoldProcess<EuclideanManifold<f64, U2>, U1, f64> for ConstantVelocityProcess {
        fn predict(
            &self,
            state: &EuclideanManifold<f64, U2>,
            _dt: f64,
            _control: Option<&OVector<f64, U1>>,
        ) -> EuclideanManifold<f64, U2> {
            EuclideanManifold::new(self.transition * state.as_vector())
        }
    }

    #[derive(Clone)]
    struct PositionMeasurement;

    impl ManifoldMeasurement<EuclideanManifold<f64, U2>, U2, U1, f64> for PositionMeasurement {
        fn measure(&self, state: &EuclideanManifold<f64, U2>) -> OVector<f64, U1> {
            Vector1::new(state.as_vector()[0])
        }

        fn residual(
            &self,
            predicted: &OVector<f64, U1>,
            measured: &OVector<f64, U1>,
        ) -> OVector<f64, U1> {
            measured - predicted
        }
    }

    #[test]
    fn predict_and_update_euclidean_state() {
        type Nominal = EuclideanManifold<f64, U2>;
        let initial_state = Nominal::new(Vector2::new(0.0, 1.0));
        let initial_cov = Matrix2::identity() * 0.1;
        let process_noise = Matrix2::identity() * 0.01;
        let measurement_noise = Matrix1::new(0.1);

        let sigma_gen = MerweScaledSigmaPoints::<f64>::new(0.5, 2.0, 0.0);
        let weights = sigma_gen.weights::<U2>();

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            initial_cov,
            ConstantVelocityProcess::new(0.1),
            process_noise,
            PositionMeasurement,
            measurement_noise,
            sigma_gen,
            weights,
        );

        ukf.predict(0.1, None).unwrap();
        let measurement = Vector1::new(0.1);
        ukf.update(&measurement).unwrap();
        let (state, _) = ukf.state_with_covariance();

        assert!((state.as_vector()[0] - 0.1).abs() < 1.0);
    }
}
