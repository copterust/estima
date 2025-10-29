use core::f64::consts::PI;

use estima::manifold::{
    composite::CompositeManifold, euclidean::EuclideanManifold, quaternion::UnitQuaternionManifold,
    ManifoldMeasurement, ManifoldProcess,
};
use estima::sigma_points::MerweScaledSigmaPoints;
use estima::UnscentedKalmanFilter;
use nalgebra::{Matrix6, UnitQuaternion, Vector3, Vector6, U3, U6};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type AttitudeManifold = UnitQuaternionManifold<f64>;
type BiasManifold = EuclideanManifold<f64, U3>;
type AHRSState = CompositeManifold<f64, AttitudeManifold, BiasManifold, U3, U3>;

#[cfg(feature = "rerun")]
struct Visualizer {
    #[cfg(feature = "rerun")]
    rec: rerun::RecordingStream,
}

#[cfg(feature = "rerun")]
impl Visualizer {
    fn new(name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let rec = rerun::RecordingStreamBuilder::new(name).spawn()?;
        Ok(Self { rec })
    }

    fn set_step(&self, step: i64) {
        self.rec.set_time_sequence("step", step);
        let _ = step;
    }

    fn log_attitude(
        &self,
        path: &str,
        quat: &UnitQuaternion<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rerun::{archetypes::Transform3D, datatypes::Quaternion as RerunQuaternion};
        self.rec.log(
            path,
            &Transform3D::from_rotation(RerunQuaternion::from_xyzw([
                quat.coords[0] as f32,
                quat.coords[1] as f32,
                quat.coords[2] as f32,
                quat.coords[3] as f32,
            ])),
        )?;
        Ok(())
    }

    fn log_axes(
        &self,
        path: &str,
        vectors: &[[f32; 3]; 3],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rerun::{archetypes::Arrows3D, Color};
        let colors = [
            Color::from_rgb(255, 0, 0),
            Color::from_rgb(0, 255, 0),
            Color::from_rgb(0, 0, 255),
        ];
        self.rec.log(
            path,
            &Arrows3D::from_vectors(vectors.iter().copied()).with_colors(colors),
        )?;
        Ok(())
    }
}

fn create_state(attitude: UnitQuaternion<f64>, bias: Vector3<f64>) -> AHRSState {
    CompositeManifold::new(
        UnitQuaternionManifold::new(attitude),
        EuclideanManifold::new(bias),
    )
}

fn attitude_component(state: &AHRSState) -> &UnitQuaternion<f64> {
    state.first.as_quaternion()
}

fn bias_component(state: &AHRSState) -> &Vector3<f64> {
    state.second.as_vector()
}

#[derive(Clone)]
struct GyroscopeProcess;

impl ManifoldProcess<AHRSState, U3, f64> for GyroscopeProcess {
    fn predict(&self, state: &AHRSState, dt: f64, control: Option<&Vector3<f64>>) -> AHRSState {
        let bias = bias_component(state);
        let attitude = attitude_component(state);

        if let Some(gyro) = control {
            let corrected = gyro - bias;
            let delta = UnitQuaternion::from_scaled_axis(corrected * dt);
            create_state(attitude * delta, *bias)
        } else {
            state.clone()
        }
    }
}

#[derive(Clone)]
struct AccelMagMeasurement {
    gravity_ref: Vector3<f64>,
    magnetic_ref: Vector3<f64>,
}

impl AccelMagMeasurement {
    fn new() -> Self {
        Self {
            gravity_ref: Vector3::new(0.0, 0.0, -9.81),
            magnetic_ref: Vector3::new(0.0, 1.0, 0.0),
        }
    }
}

impl ManifoldMeasurement<AHRSState, U6, U6, f64> for AccelMagMeasurement {
    fn measure(&self, state: &AHRSState) -> Vector6<f64> {
        let attitude = attitude_component(state);
        let accel = -attitude.inverse().transform_vector(&self.gravity_ref);
        let mag = attitude.inverse().transform_vector(&self.magnetic_ref);

        let accel_norm = accel.normalize();
        let mag_norm = mag.normalize();

        Vector6::new(
            accel_norm.x,
            accel_norm.y,
            accel_norm.z,
            mag_norm.x,
            mag_norm.y,
            mag_norm.z,
        )
    }

    fn residual(&self, predicted: &Vector6<f64>, measured: &Vector6<f64>) -> Vector6<f64> {
        let pred_accel = Vector3::new(predicted[0], predicted[1], predicted[2]).normalize();
        let pred_mag = Vector3::new(predicted[3], predicted[4], predicted[5]).normalize();

        let meas_accel = Vector3::new(measured[0], measured[1], measured[2]).normalize();
        let meas_mag = Vector3::new(measured[3], measured[4], measured[5]).normalize();

        let accel_residual = meas_accel.cross(&pred_accel);
        let mag_residual = meas_mag.cross(&pred_mag);

        Vector6::new(
            accel_residual.x,
            accel_residual.y,
            accel_residual.z,
            mag_residual.x,
            mag_residual.y,
            mag_residual.z,
        )
    }

    fn innovation(&self, measured: &Vector6<f64>, predicted_mean: &Vector6<f64>) -> Vector6<f64> {
        self.residual(predicted_mean, measured)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "rerun")]
    let viz = Visualizer::new("estima_ahrs_ukf")?;

    let dt: f64 = 0.01;
    let gyro_noise: f64 = 0.01;
    let accel_noise: f64 = 0.1;
    let mag_noise: f64 = 0.1;

    let true_attitude_initial = UnitQuaternion::from_euler_angles(0.1, -0.1, 0.05);
    let true_bias = Vector3::new(0.01, -0.01, 0.005);

    let initial_attitude_error = UnitQuaternion::from_euler_angles(0.1, 0.2, -0.15);
    let initial_state = create_state(
        true_attitude_initial * initial_attitude_error,
        Vector3::zeros(),
    );

    let initial_covariance = Matrix6::<f64>::identity() * 0.2;

    let mut process_noise = Matrix6::<f64>::zeros();
    process_noise
        .fixed_view_mut::<3, 3>(0, 0)
        .fill_diagonal(0.05f64.powi(2));
    process_noise
        .fixed_view_mut::<3, 3>(3, 3)
        .fill_diagonal(gyro_noise.powi(2) * dt);

    let mut measurement_noise = Matrix6::<f64>::zeros();
    measurement_noise
        .fixed_view_mut::<3, 3>(0, 0)
        .fill_diagonal(accel_noise.powi(2));
    measurement_noise
        .fixed_view_mut::<3, 3>(3, 3)
        .fill_diagonal(mag_noise.powi(2));

    let sigma_gen = MerweScaledSigmaPoints::new(0.5, 2.0, 0.0);
    let weights = sigma_gen.weights::<U6>();

    let measurement_model = AccelMagMeasurement::new();
    let mut ukf = UnscentedKalmanFilter::new(
        initial_state,
        initial_covariance,
        GyroscopeProcess,
        process_noise,
        measurement_model.clone(),
        measurement_noise,
        sigma_gen,
        weights,
    )
    .with_regularization_factor(1e-3);

    let mut rng = StdRng::seed_from_u64(2024);
    let mut true_attitude = true_attitude_initial;
    let gyro_bias = true_bias;

    #[cfg(feature = "rerun")]
    let body_axes = [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];

    let swing_period_steps = 200;
    let total_steps = swing_period_steps * 6;
    let swing_amplitude = PI;
    let omega_mag = swing_amplitude / (swing_period_steps as f64 / 2.0 * dt);

    for step in 0..total_steps {
        #[cfg(feature = "rerun")]
        viz.set_step(step as i64);

        let segment = step / swing_period_steps;
        let phase = step % swing_period_steps;
        let direction = if phase < swing_period_steps / 2 {
            1.0
        } else {
            -1.0
        };

        let angular_velocity_world = match segment {
            0 => Vector3::new(direction * omega_mag, 0.0, 0.0),
            1 => Vector3::new(0.0, direction * omega_mag, 0.0),
            2 => Vector3::new(0.0, 0.0, direction * omega_mag),
            _ => Vector3::zeros(),
        };

        true_attitude =
            UnitQuaternion::from_scaled_axis(angular_velocity_world * dt) * true_attitude;

        let gyro_measurement = true_attitude
            .inverse()
            .transform_vector(&angular_velocity_world)
            + gyro_bias;
        let accel_body = -true_attitude
            .inverse()
            .transform_vector(&measurement_model.gravity_ref);
        let mag_body = true_attitude
            .inverse()
            .transform_vector(&measurement_model.magnetic_ref);

        let accel_meas = accel_body.normalize()
            + Vector3::new(
                rng.random_range(-accel_noise..accel_noise),
                rng.random_range(-accel_noise..accel_noise),
                rng.random_range(-accel_noise..accel_noise),
            ) * 0.05;
        let mag_meas = mag_body.normalize()
            + Vector3::new(
                rng.random_range(-mag_noise..mag_noise),
                rng.random_range(-mag_noise..mag_noise),
                rng.random_range(-mag_noise..mag_noise),
            ) * 0.02;

        let measurement = Vector6::new(
            accel_meas.x,
            accel_meas.y,
            accel_meas.z,
            mag_meas.x,
            mag_meas.y,
            mag_meas.z,
        );

        ukf.predict(dt, Some(&gyro_measurement))
            .map_err(|e| format!("UKF predict failed: {:?}", e))?;
        ukf.update(&measurement)
            .map_err(|e| format!("UKF update failed: {:?}", e))?;

        let estimate = ukf.nominal_state();
        let estimated_attitude = attitude_component(estimate);
        let estimated_bias = bias_component(estimate);

        let attitude_error = true_attitude.inverse() * estimated_attitude;
        let error_angle = 2.0 * attitude_error.w.abs().acos();
        let bias_error = (gyro_bias - estimated_bias).norm();

        if step % 100 == 0 {
            println!(
                "step {:04}: attitude error = {:6.3}°, bias error = {:8.6} rad/s",
                step,
                error_angle.to_degrees(),
                bias_error
            );
        }

        #[cfg(feature = "rerun")]
        {
            let true_axes = body_axes.map(|axis| {
                let v = true_attitude.transform_vector(&axis);
                [v.x as f32, v.y as f32, v.z as f32]
            });
            let est_axes = body_axes.map(|axis| {
                let v = estimated_attitude.transform_vector(&axis);
                [v.x as f32, v.y as f32, v.z as f32]
            });

            viz.log_attitude("attitude/true", &true_attitude)?;
            viz.log_attitude("attitude/estimated", estimated_attitude)?;
            viz.log_axes("orientation/true_axes", &true_axes)?;
            viz.log_axes("orientation/estimated_axes", &est_axes)?;
        }
    }

    Ok(())
}
