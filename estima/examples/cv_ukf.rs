use estima::manifold::{euclidean::EuclideanManifold, ManifoldMeasurement, ManifoldProcess};
use estima::sigma_points::MerweScaledSigmaPoints;
use estima::UnscentedKalmanFilter;
use nalgebra::{Matrix1, Matrix2, OVector, Vector1, Vector2, U1, U2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type CVState = EuclideanManifold<f64, U2>;

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

fn main() {
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

    let mut rng = StdRng::seed_from_u64(42);
    let mut true_pos = 0.0_f64;
    let true_vel = 1.0_f64;

    for step in 0..50 {
        true_pos += true_vel * dt;
        let noisy_measurement = Vector1::new(true_pos + rng.gen_range(-1.0..1.0) * r.sqrt());

        ukf.predict(dt, None).expect("prediction failed");
        ukf.update(&noisy_measurement).expect("update failed");

        let (state, cov) = ukf.state_with_covariance();
        let pos = state.as_vector()[0];
        let vel = state.as_vector()[1];
        let pos_std = cov[(0, 0)].sqrt();
        let vel_std = cov[(1, 1)].sqrt();

        if step % 10 == 0 || step == 49 {
            println!(
                "step {:02}: pos {:.3}±{:.3}, vel {:.3}±{:.3}, true pos {:.3}",
                step, pos, pos_std, vel, vel_std, true_pos
            );
        }
    }
}
