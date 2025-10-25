use estima::filter::UnscentedKalmanFilter;
use estima::manifold::{euclidean::EuclideanManifold, ManifoldMeasurement, ManifoldProcess};
use estima::sigma_points::MerweScaledSigmaPoints;
use estima::sr_ukf::UTWeights;
use nalgebra::{DimName, Matrix1, Matrix2, OVector, Vector2, U1, U2};
use ndarray::Array;
use rand::Rng;
use rerun::{
    archetypes::{Scalars, Tensor},
    RecordingStreamBuilder,
};

/// Complete CV state: 2D vector in Euclidean space R²
type CVState = EuclideanManifold<f64, U2>;

// 2. Define the process model
struct CVProcessModel {
    transition_matrix: Matrix2<f64>,
}

impl CVProcessModel {
    pub fn new(dt: f64) -> Self {
        Self {
            transition_matrix: Matrix2::new(1.0, dt, 0.0, 1.0),
        }
    }
}

impl ManifoldProcess<CVState, U1, f64> for CVProcessModel {
    fn predict(&self, state: &CVState, _dt: f64, _control: Option<&OVector<f64, U1>>) -> CVState {
        CVState::new(self.transition_matrix * state.as_vector())
    }
}

// 3. Define the measurement model
struct CVMeasurementModel;

impl ManifoldMeasurement<CVState, U2, U1, f64> for CVMeasurementModel {
    fn measure(&self, x: &CVState) -> OVector<f64, U1> {
        OVector::<f64, U1>::new(x.as_vector()[0])
    }

    fn residual(
        &self,
        predicted: &OVector<f64, U1>,
        measured: &OVector<f64, U1>,
    ) -> OVector<f64, U1> {
        measured - predicted
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec = RecordingStreamBuilder::new("estima_cv_ukf").spawn()?;

    // 4. System parameters
    let dt: f64 = 0.1;
    let q = 0.01;
    let r: f64 = 0.1;

    // 5. Initial state and covariance
    let initial_state = CVState::new(Vector2::new(0.0, 1.0));
    let initial_covariance_sqrt = Matrix2::identity();

    // 6. Noise matrices
    let process_noise = Matrix2::new(
        q * dt.powi(3) / 3.0,
        q * dt.powi(2) / 2.0,
        q * dt.powi(2) / 2.0,
        q * dt,
    );
    let measurement_noise = Matrix1::new(r);

    // 7. UKF parameters
    let alpha = 0.001;
    let beta = 2.0;
    let kappa = 0.0;

    // 8. Create the UKF
    let mut ukf: UnscentedKalmanFilter<
        CVState,
        CVProcessModel,
        CVMeasurementModel,
        U2, // TangentDim
        U1, // ControlDim
        U1, // MeasDim
        MerweScaledSigmaPoints<f64>,
        f64,
    > = UnscentedKalmanFilter::new(
        initial_state,
        initial_covariance_sqrt,
        CVProcessModel::new(dt),
        process_noise,
        CVMeasurementModel,
        measurement_noise,
        MerweScaledSigmaPoints::<f64>::new(alpha, beta, kappa),
        UTWeights::new(U2::dim(), alpha, beta, kappa),
    );

    // 9. Generate some measurements
    let mut measurements = Vec::new();
    let mut true_positions = Vec::new();
    let mut true_pos = 0.0;
    let true_vel = 1.0;
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        true_pos += true_vel * dt;
        true_positions.push(true_pos);
        measurements.push(OVector::<f64, U1>::new(
            true_pos + rng.gen_range(-1.0..1.0) * r.sqrt(),
        ));
    }

    // 10. Run the filter and log data to rerun
    for (i, (measurement, &true_pos)) in measurements.iter().zip(true_positions.iter()).enumerate()
    {
        rec.set_time_sequence("step", i as i64);

        // Log ground truth and measurement
        rec.log("true/position", &Scalars::new([true_pos]))?;
        rec.log("true/velocity", &Scalars::new([true_vel]))?;
        rec.log("measurement/position", &Scalars::new([measurement[0]]))?;

        // Predict
        ukf.predict(dt, None).unwrap();
        let (state_predicted, cov_predicted) = ukf.state_with_covariance();
        let pos_std_dev_predicted = cov_predicted[(0, 0)].sqrt();
        let vel_std_dev_predicted = cov_predicted[(1, 1)].sqrt();

        println!(
            "Step {}: Predicted Pos = {:.4}, Vel = {:.4}, Pos Std Dev = {:.4}, Vel Std Dev = {:.4}",
            i,
            state_predicted.as_vector()[0],
            state_predicted.as_vector()[1],
            pos_std_dev_predicted,
            vel_std_dev_predicted
        );

        // Log predicted state
        rec.log(
            "predicted/position",
            &Scalars::new([state_predicted.as_vector()[0]]),
        )?;
        rec.log(
            "predicted/velocity",
            &Scalars::new([state_predicted.as_vector()[1]]),
        )?;
        rec.log(
            "predicted/position_std_dev",
            &Scalars::new([pos_std_dev_predicted]),
        )?;
        rec.log(
            "predicted/velocity_std_dev",
            &Scalars::new([vel_std_dev_predicted]),
        )?;

        // Update
        ukf.update(measurement).unwrap();
        let (state_updated, cov_updated) = ukf.state_with_covariance();
        let pos_std_dev_updated = cov_updated[(0, 0)].sqrt();
        let vel_std_dev_updated = cov_updated[(1, 1)].sqrt();

        println!(
            "Step {}: Updated Pos = {:.4}, Vel = {:.4}, Pos Std Dev = {:.4}, Vel Std Dev = {:.4}",
            i,
            state_updated.as_vector()[0],
            state_updated.as_vector()[1],
            pos_std_dev_updated,
            vel_std_dev_updated
        );
        // Log errors
        let pos_error = state_updated.as_vector()[0] - true_pos;
        let vel_error = state_updated.as_vector()[1] - true_vel;
        rec.log("error/position", &Scalars::new([pos_error]))?;
        rec.log("error/velocity", &Scalars::new([vel_error]))?;

        // Log covariance matrix
        let cov_array = Array::from_shape_vec((2, 2), cov_updated.transpose().as_slice().to_vec())?;
        let tensor = Tensor::try_from(cov_array)?.with_dim_names(["rows", "cols"]);
        rec.log("updated/covariance", &tensor)?;
    }

    Ok(())
}
