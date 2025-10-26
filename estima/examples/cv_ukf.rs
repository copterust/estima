use estima::manifold::{euclidean::EuclideanManifold, ManifoldMeasurement, ManifoldProcess};
use estima::sigma_points::MerweScaledSigmaPoints;
use estima::UnscentedKalmanFilter;
use nalgebra::{Matrix1, Matrix2, OVector, Vector1, Vector2, U1, U2};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::error::Error;

#[cfg(feature = "rerun")]
use rerun::{
    archetypes::{LineStrips2D, Points2D},
    Color, RecordingStreamBuilder,
};

type CVState = EuclideanManifold<f64, U2>;

#[cfg(feature = "rerun")]
struct Visualizer {
    rec: rerun::RecordingStream,
    times: Vec<f64>,
    pos_truth: Vec<f64>,
    pos_mean: Vec<f64>,
    pos_upper: Vec<f64>,
    pos_lower: Vec<f64>,
    pos_measurements: Vec<[f32; 2]>,
    vel_truth: Vec<f64>,
    vel_mean: Vec<f64>,
    vel_upper: Vec<f64>,
    vel_lower: Vec<f64>,
}

#[cfg(feature = "rerun")]
impl Visualizer {
    const COLOR_TRUE: [u8; 3] = [60, 179, 113];
    const COLOR_MEAN: [u8; 3] = [30, 144, 255];
    const COLOR_CONFIDENCE: [u8; 3] = [220, 20, 60];
    const COLOR_MEASUREMENT: [u8; 3] = [255, 140, 0];

    fn new(name: &str) -> Result<Self, Box<dyn Error>> {
        let rec = RecordingStreamBuilder::new(name).spawn()?;
        Ok(Self {
            rec,
            times: Vec::new(),
            pos_truth: Vec::new(),
            pos_mean: Vec::new(),
            pos_upper: Vec::new(),
            pos_lower: Vec::new(),
            pos_measurements: Vec::new(),
            vel_truth: Vec::new(),
            vel_mean: Vec::new(),
            vel_upper: Vec::new(),
            vel_lower: Vec::new(),
        })
    }

    fn set_step(&self, step: i64) {
        self.rec.set_time_sequence("step", step);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_state(
        &mut self,
        time: f64,
        true_pos: f64,
        true_vel: f64,
        measurement_pos: f64,
        mean_pos: f64,
        mean_vel: f64,
        pos_std: f64,
        vel_std: f64,
    ) -> Result<(), Box<dyn Error>> {
        const CONF_95: f64 = 1.96;

        self.times.push(time);
        self.pos_truth.push(true_pos);
        self.pos_mean.push(mean_pos);
        self.pos_upper.push(mean_pos + CONF_95 * pos_std);
        self.pos_lower.push(mean_pos - CONF_95 * pos_std);
        self.pos_measurements
            .push([time as f32, measurement_pos as f32]);

        self.vel_truth.push(true_vel);
        self.vel_mean.push(mean_vel);
        self.vel_upper.push(mean_vel + CONF_95 * vel_std);
        self.vel_lower.push(mean_vel - CONF_95 * vel_std);

        self.log_line(
            "position_plot/true",
            &self.pos_truth,
            Self::COLOR_TRUE,
            0.008,
        )?;
        self.log_line("position_plot/mean", &self.pos_mean, Self::COLOR_MEAN, 0.01)?;
        self.log_line(
            "position_plot/confidence_upper",
            &self.pos_upper,
            Self::COLOR_CONFIDENCE,
            0.006,
        )?;
        self.log_line(
            "position_plot/confidence_lower",
            &self.pos_lower,
            Self::COLOR_CONFIDENCE,
            0.006,
        )?;
        self.log_points(
            "position_plot/measurements",
            &self.pos_measurements,
            Self::COLOR_MEASUREMENT,
            0.012,
        )?;

        self.log_line(
            "velocity_plot/true",
            &self.vel_truth,
            Self::COLOR_TRUE,
            0.008,
        )?;
        self.log_line("velocity_plot/mean", &self.vel_mean, Self::COLOR_MEAN, 0.01)?;
        self.log_line(
            "velocity_plot/confidence_upper",
            &self.vel_upper,
            Self::COLOR_CONFIDENCE,
            0.006,
        )?;
        self.log_line(
            "velocity_plot/confidence_lower",
            &self.vel_lower,
            Self::COLOR_CONFIDENCE,
            0.006,
        )?;
        Ok(())
    }

    fn log_line(
        &self,
        path: &str,
        values: &[f64],
        color: [u8; 3],
        radius: f32,
    ) -> Result<(), Box<dyn Error>> {
        if self.times.len() < 2 || values.len() != self.times.len() {
            return Ok(());
        }

        let points: Vec<[f32; 2]> = self
            .times
            .iter()
            .zip(values.iter())
            .map(|(x, y)| [*x as f32, *y as f32])
            .collect();

        let line = LineStrips2D::new([points])
            .with_colors([Color::from_rgb(color[0], color[1], color[2])])
            .with_radii([radius]);
        self.rec.log(path, &line)?;
        Ok(())
    }

    fn log_points(
        &self,
        path: &str,
        points: &[[f32; 2]],
        color: [u8; 3],
        radius: f32,
    ) -> Result<(), Box<dyn Error>> {
        if points.is_empty() {
            return Ok(());
        }

        let points = Points2D::new(points.to_vec())
            .with_colors([Color::from_rgb(color[0], color[1], color[2])])
            .with_radii([radius]);
        self.rec.log(path, &points)?;
        Ok(())
    }
}

#[derive(Clone)]
struct CVProcessModel {
    transition: Matrix2<f64>,
}

impl CVProcessModel {
    fn new(dt: f64) -> Self {
        Self {
            transition: Matrix2::new(1.0, dt, 0.0, 1.0),
        }
    }
}

impl ManifoldProcess<CVState, U1, f64> for CVProcessModel {
    fn predict(&self, state: &CVState, _dt: f64, _control: Option<&OVector<f64, U1>>) -> CVState {
        CVState::new(self.transition * state.as_vector())
    }
}

#[derive(Clone)]
struct CVMeasurementModel;

impl ManifoldMeasurement<CVState, U2, U1, f64> for CVMeasurementModel {
    fn measure(&self, state: &CVState) -> OVector<f64, U1> {
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

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "rerun")]
    let mut viz = Visualizer::new("estima_cv_ukf")?;

    let dt: f64 = 0.1;
    let q: f64 = 0.01;
    let r: f64 = 0.1;

    let initial_state = CVState::new(Vector2::new(0.0, 1.0));
    let initial_covariance = Matrix2::identity() * 0.1;

    let process_noise = Matrix2::new(
        q * dt.powi(3) / 3.0,
        q * dt.powi(2) / 2.0,
        q * dt.powi(2) / 2.0,
        q * dt,
    );
    let measurement_noise = Matrix1::new(r);

    let sigma_gen = MerweScaledSigmaPoints::new(0.5, 2.0, 0.0);
    let weights = sigma_gen.weights::<U2>();

    let mut ukf: UnscentedKalmanFilter<
        CVState,
        CVProcessModel,
        CVMeasurementModel,
        U2,
        U1,
        U1,
        MerweScaledSigmaPoints<f64>,
        f64,
    > = UnscentedKalmanFilter::new(
        initial_state,
        initial_covariance,
        CVProcessModel::new(dt),
        process_noise,
        CVMeasurementModel,
        measurement_noise,
        sigma_gen,
        weights,
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut true_pos = 0.0_f64;
    let true_vel = 1.0_f64;

    let total_steps = 50;
    for step in 0..total_steps {
        #[cfg(feature = "rerun")]
        viz.set_step(step as i64);

        true_pos += true_vel * dt;
        let noisy_measurement =
            Vector1::new(true_pos + rng.sample::<f64, _>(StandardNormal) * r.sqrt());
        ukf.predict(dt, None)
            .map_err(|e| format!("UKF predict failed: {:?}", e))?;
        ukf.update(&noisy_measurement)
            .map_err(|e| format!("UKF update failed: {:?}", e))?;

        let (state_updated, cov_updated) = ukf.state_with_covariance();
        let pos = state_updated.as_vector()[0];
        let vel = state_updated.as_vector()[1];
        let pos_std = cov_updated[(0, 0)].sqrt();
        let vel_std = cov_updated[(1, 1)].sqrt();

        #[cfg(feature = "rerun")]
        viz.record_state(
            step as f64 * dt,
            true_pos,
            true_vel,
            noisy_measurement[0],
            pos,
            vel,
            pos_std,
            vel_std,
        )?;

        if step % 10 == 0 || step + 1 == total_steps {
            println!(
                "step {:02}: pos {:.3}±{:.3}, vel {:.3}±{:.3}, true pos {:.3}",
                step, pos, pos_std, vel, vel_std, true_pos
            );
        }
    }

    Ok(())
}
