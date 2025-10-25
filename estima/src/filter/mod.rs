use std::marker::PhantomData;

use nalgebra::allocator::Allocator;
use nalgebra::DefaultAllocator;
use nalgebra::{Cholesky, Const, DimName, OMatrix, OVector, RealField};

use crate::kf::averaging::{LinearAveraging, WeightedMean};
use crate::kf::error::UkfError;
use crate::manifold::{InitialGuess, Manifold, ManifoldMeasurement, ManifoldProcess, MeanError};
use crate::sigma_points::SigmaPointsInPlace;
use crate::sr_ukf::engine::UKFEngine;
use crate::sr_ukf::UTWeights;

/// A unified Unscented Kalman Filter for states living on manifolds.
///
/// This filter separates the nominal state (living on the manifold) from the
/// error state (living in the tangent space). The error state is used for
/// all covariance computations and is periodically "injected" into the nominal
/// state via the manifold's retract operation.
///
/// This implementation is general and works for both manifold and standard Euclidean states.
/// For Euclidean states, the `Manifold` trait should be implemented such that `retract`
/// is vector addition and `local` is vector subtraction.
///
/// # Type Parameters
/// * `Nominal` - The nominal state type (must implement `Manifold`)
/// * `Process` - Process model for predicting nominal states
/// * `Measurement` - Measurement model for nominal states
/// * `TangentDim` - Dimension of the manifold's tangent space
/// * `ControlDim` - Dimension of control inputs
/// * `MeasDim` - Dimension of measurements
/// * `SigmaGen` - Sigma point generator
/// * `T` - Scalar type (usually `f64` or `f32`)
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
    /// Nominal state on the manifold
    nominal_state: Nominal,
    /// Square root of the error covariance matrix in the tangent space
    error_covariance_sqrt: Cholesky<T, TangentDim>,
    /// Process model for the nominal state
    process_model: Process,
    /// Process noise covariance (in tangent space)
    process_noise_cov: OMatrix<T, TangentDim, TangentDim>,
    /// Measurement model for the nominal state
    measurement_model: Measurement,
    /// Measurement noise covariance
    measurement_noise_cov: OMatrix<T, MeasDim, MeasDim>,
    /// Sigma point generator
    sigma_generator: SigmaGen,
    /// UKF engine for core calculations
    engine: UKFEngine<TangentDim, MeasDim, SigmaGen::SigmaCount, T>,
    /// Buffer for nominal sigma points
    nominal_sigmas: Vec<Nominal>,
    /// Buffer for predicted sigma points
    predicted_sigmas: Vec<Nominal>,
    /// Weighted mean calculator for measurements
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
    ///
    /// # Arguments
    /// * `initial_nominal` - Initial nominal state on the manifold
    /// * `initial_error_covariance_sqrt` - Initial error covariance square root (tangent space)
    /// * `process_model` - Process model for state prediction
    /// * `process_noise_cov` - Process noise covariance (tangent space)
    /// * `measurement_model` - Measurement model
    /// * `measurement_noise_cov` - Measurement noise covariance
    /// * `sigma_generator` - Sigma point generator
    /// * `weights` - UKF weights
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
                .expect("Initial error covariance must be positive definite"),
            process_model,
            process_noise_cov,
            measurement_model,
            measurement_noise_cov,
            sigma_generator,
            engine,
            nominal_sigmas: vec![initial_nominal.clone(); n_sigmas],
            predicted_sigmas: vec![initial_nominal.clone(); n_sigmas],
            weighted_mean: LinearAveraging,
            _phantom_control: PhantomData,
        }
    }

    /// Set the regularization factor for numerical stability.
    pub fn with_regularization_factor(mut self, factor: T) -> Self {
        self.engine.regularization_factor = factor;
        self
    }

    /// Get the current nominal state.
    pub fn nominal_state(&self) -> &Nominal {
        &self.nominal_state
    }

    /// Get the current error covariance matrix.
    pub fn error_covariance(&self) -> OMatrix<T, TangentDim, TangentDim> {
        let l = self.error_covariance_sqrt.l();
        &l * l.transpose()
    }

    /// Predict the state forward in time.
    ///
    /// # Arguments
    /// * `dt` - Time step
    /// * `control` - Optional control input
    pub fn predict(
        &mut self,
        dt: T,
        control: Option<&OVector<T, ControlDim>>,
    ) -> Result<(), UkfError> {
        self.sigma_generator.generate_into(
            &OVector::zeros_generic(TangentDim::name(), Const::<1>),
            &self.error_covariance_sqrt,
            &mut self.engine.sigma_points,
            &mut self.engine.w_mean_scratch,
            &mut self.engine.w_covar_scratch,
        );

        // Map tangent sigma points to nominal states on manifold
        for (i, nominal_sigma) in self.nominal_sigmas.iter_mut().enumerate() {
            let tangent_sigma = self.engine.sigma_points.column(i);
            *nominal_sigma = self.nominal_state.retract(&tangent_sigma.into_owned());
        }

        // Predict each nominal sigma point through process model
        for (i, predicted) in self.predicted_sigmas.iter_mut().enumerate() {
            *predicted = self
                .process_model
                .predict(&self.nominal_sigmas[i], dt, control);
        }

        // Compute predicted nominal state (weighted mean on manifold)
        let weighted_mean_result = Nominal::weighted_mean(
            &self.predicted_sigmas,
            self.engine.weights.w_mean.as_slice(),
            T::from_subset(&1e-9),
            InitialGuess::MaxWeight,
            100,
        );

        self.nominal_state = match weighted_mean_result {
            Ok(mean) => mean,
            Err(e) => {
                match e {
                    MeanError::NotConverged => {
                        // Fallback: use first sigma point if weighted mean fails to converge
                        // This is a design choice to allow the filter to continue
                        self.predicted_sigmas[0].clone()
                    }
                    MeanError::EmptyInput
                    | MeanError::NoPositiveWeights
                    | MeanError::InvalidTolerance
                    | MeanError::IndexOutOfBounds => return Err(UkfError::MeanComputationFailed),
                }
            }
        };

        // Compute predicted error covariance in tangent space
        // Use batch operation for better performance
        Nominal::batch_local_into_matrix(
            &self.nominal_state,
            &self.predicted_sigmas,
            &mut self.engine.transformed,
        );

        // The mean of the transformed sigma points in the tangent space is not necessarily zero.
        // We must compute it to correctly calculate the covariance deviations.
        self.weighted_mean.weighted_mean(
            &self.engine.transformed,
            &self.engine.weights.w_mean,
            &mut self.engine.x_buffer,
        );

        // Call engine's predict_covariance
        self.error_covariance_sqrt = self.engine.predict_covariance(&self.process_noise_cov)?;
        Ok(())
    }

    /// Update the state with a measurement.
    ///
    /// # Arguments
    /// * `measurement` - The observed measurement
    pub fn update(&mut self, measurement: &OVector<T, MeasDim>) -> Result<(), UkfError> {
        // Generate sigma points in tangent space
        self.sigma_generator.generate_into(
            &OVector::zeros_generic(TangentDim::name(), Const::<1>),
            &self.error_covariance_sqrt,
            &mut self.engine.sigma_points,
            &mut self.engine.w_mean_scratch,
            &mut self.engine.w_covar_scratch,
        );

        // Map tangent sigma points to nominal states on manifold
        for (i, nominal_sigma) in self.nominal_sigmas.iter_mut().enumerate() {
            let tangent_sigma = self.engine.sigma_points.column(i);
            *nominal_sigma = self.nominal_state.retract(&tangent_sigma.into_owned());
        }

        // Predict measurements for each sigma point
        for i in 0..self.nominal_sigmas.len() {
            let z_sigma = self.measurement_model.measure(&self.nominal_sigmas[i]);
            self.engine.z_sigma_buffer.column_mut(i).copy_from(&z_sigma);
        }

        // Compute predicted measurement mean (z_pred)
        let mut z_pred = OVector::<T, MeasDim>::zeros();
        self.weighted_mean.weighted_mean(
            &self.engine.z_sigma_buffer,
            &self.engine.weights.w_mean,
            &mut z_pred,
        );

        // Compute innovation using the true predicted mean
        let innovation = self.measurement_model.innovation(measurement, &z_pred);

        // Manually compute measurement deviations using the proper residual for the manifold.
        // This is critical because the UKFEngine assumes Euclidean vector subtraction, which is
        // incorrect for measurements like unit vectors where the residual is a cross product.
        for i in 0..self.nominal_sigmas.len() {
            let z_sigma = self.engine.z_sigma_buffer.column(i);
            let residual = self
                .measurement_model
                .residual(&z_pred, &z_sigma.into_owned());
            self.engine
                .measurement_deviations
                .column_mut(i)
                .copy_from(&residual);
        }

        // The state deviations for the update step are simply the sigma points, as they are generated
        // around a zero-mean tangent space.
        self.engine
            .state_deviations
            .copy_from(&self.engine.sigma_points);

        let (kalman_gain, _z_pred, new_error_covariance_sqrt) = self.engine.update(
            &self.error_covariance_sqrt,
            &self.measurement_noise_cov,
            &self.weighted_mean,
        )?;

        let error_update = &kalman_gain * innovation;

        self.nominal_state = self.nominal_state.retract(&error_update);
        self.error_covariance_sqrt = new_error_covariance_sqrt;

        Ok(())
    }

    /// Get the current state estimate with covariance.
    pub fn state_with_covariance(&self) -> (Nominal, OMatrix<T, TangentDim, TangentDim>) {
        (self.nominal_state.clone(), self.error_covariance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::quaternion::UnitQuaternionManifold;
    use crate::sigma_points::MerweScaledSigmaPoints;
    use nalgebra::{Matrix3, UnitQuaternion, Vector3, U3};

    // Simple test implementations for process and measurement models
    struct IdentityProcess;
    impl ManifoldProcess<UnitQuaternionManifold<f64>, U3, f64> for IdentityProcess {
        fn predict(
            &self,
            state: &UnitQuaternionManifold<f64>,
            _dt: f64,
            control: Option<&Vector3<f64>>,
        ) -> UnitQuaternionManifold<f64> {
            if let Some(angular_velocity) = control {
                let delta_rotation = UnitQuaternion::from_scaled_axis(*angular_velocity);
                UnitQuaternionManifold::new(delta_rotation * *state.as_quaternion())
            } else {
                *state
            }
        }
    }

    struct GravityMeasurement;
    impl ManifoldMeasurement<UnitQuaternionManifold<f64>, U3, U3, f64> for GravityMeasurement {
        fn measure(&self, state: &UnitQuaternionManifold<f64>) -> Vector3<f64> {
            // Transform gravity vector from world to body frame
            let gravity_world = Vector3::new(0.0, 0.0, -9.81);
            state
                .as_quaternion()
                .inverse()
                .transform_vector(&gravity_world)
        }
        fn residual(
            &self,
            predicted: &OVector<f64, U3>,
            measured: &OVector<f64, U3>,
        ) -> OVector<f64, U3> {
            measured - predicted
        }
    }

    #[test]
    fn test_unscented_kalman_filter_construction() {
        let initial_state = UnitQuaternionManifold::new(UnitQuaternion::identity());
        let initial_cov = Matrix3::identity() * 0.1;
        let process_noise = Matrix3::identity() * 0.01;
        let measurement_noise = Matrix3::identity() * 0.1;

        let sigma_gen = MerweScaledSigmaPoints::new(0.5, 2.0, 0.0);
        let weights = UTWeights::from_merwe_scaled_parameters(3, 0.5, 2.0, 0.0);

        let _ukf = UnscentedKalmanFilter::new(
            initial_state,
            initial_cov,
            IdentityProcess,
            process_noise,
            GravityMeasurement,
            measurement_noise,
            sigma_gen,
            weights,
        );

        // If we get here without panicking, construction succeeded
    }

    #[test]
    fn test_predict_step() {
        let initial_state = UnitQuaternionManifold::new(UnitQuaternion::identity());
        let initial_cov = Matrix3::identity() * 0.1;
        let process_noise = Matrix3::identity() * 0.01;
        let measurement_noise = Matrix3::identity() * 0.1;

        let sigma_gen = MerweScaledSigmaPoints::new(0.5, 2.0, 0.0);
        let weights = UTWeights::from_merwe_scaled_parameters(3, 0.5, 2.0, 0.0);

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            initial_cov,
            IdentityProcess,
            process_noise,
            GravityMeasurement,
            measurement_noise,
            sigma_gen,
            weights,
        );

        let angular_velocity = Vector3::new(0.1, 0.0, 0.0);
        ukf.predict(0.01, Some(&angular_velocity));

        // Check that state has rotated approximately as expected
        let predicted_state = ukf.nominal_state();
        let expected_rotation = UnitQuaternion::from_scaled_axis(angular_velocity * 0.01);
        let actual_rotation = *predicted_state.as_quaternion();

        // Check rotation is approximately correct (within some tolerance)
        let rotation_error = (expected_rotation.inverse() * actual_rotation)
            .scaled_axis()
            .norm();
        assert!(
            rotation_error < 0.1,
            "Rotation error too large: {}",
            rotation_error
        );
    }
}
