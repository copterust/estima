use estima::kf::traits::{MeasurementModel, ProcessModel, State};
use estima::sigma_points::MerweScaled;
use estima::sr_ukf::{SquareRootUKF, UTWeights};
use nalgebra::{DimName, Matrix1, Matrix2, OVector, Vector2, U1, U2};
use ndarray::Array;
use rand::Rng;
use rerun::{
    archetypes::{Scalars, Tensor},
    RecordingStreamBuilder,
};

// 1. Define the state
#[derive(Clone, Debug)]
struct CVState(Vector2<f64>);

impl State<U2, f64> for CVState {
    fn into_vector(self) -> Vector2<f64> {
        self.0
    }

    fn write_into(&self, buf: &mut Vector2<f64>) {
        buf.copy_from(&self.0);
    }

    fn from_vector(v: Vector2<f64>) -> Self {
        CVState(v)
    }
}

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

impl ProcessModel<U2, U1, f64> for CVProcessModel {
    fn predict<S>(
        &self,
        state: &nalgebra::Matrix<f64, U2, nalgebra::Const<1>, S>,
        _dt: f64,
        _control: Option<&OVector<f64, U1>>,
    ) -> Vector2<f64>
    where
        S: nalgebra::storage::Storage<f64, U2, nalgebra::Const<1>>,
    {
        self.transition_matrix * state
    }
}

// 3. Define the measurement model
struct CVMeasurementModel;

impl MeasurementModel<U2, U1, f64> for CVMeasurementModel {
    fn measure(&self, x: &OVector<f64, U2>) -> OVector<f64, U1> {
        OVector::<f64, U1>::new(x[0])
    }

    fn residual(&self, z_pred: &OVector<f64, U1>, z_meas: &OVector<f64, U1>) -> OVector<f64, U1> {
        z_meas - z_pred
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec = RecordingStreamBuilder::new("estima_cv_ukf").spawn()?;

    // 4. System parameters
    let dt: f64 = 0.1;
    let q = 0.01;
    let r = 0.1;

    // 5. Initial state and covariance
    let initial_state = CVState(Vector2::new(0.0, 1.0));
    let initial_covariance_sqrt = Matrix2::identity();

    // 6. Noise matrices
    let process_noise_sqrt = Matrix2::new(
        q * dt.powi(3) / 3.0,
        q * dt.powi(2) / 2.0,
        q * dt.powi(2) / 2.0,
        q * dt,
    )
    .cholesky()
    .unwrap()
    .l();
    let measurement_noise_sqrt = Matrix1::new(r).cholesky().unwrap().l();

    // 7. UKF parameters
    let alpha = 0.001;
    let beta = 2.0;
    let kappa = 0.0;

    // 8. Create the UKF
    let mut ukf = SquareRootUKF::new(
        initial_state,
        initial_covariance_sqrt,
        CVProcessModel::new(dt),
        process_noise_sqrt,
        CVMeasurementModel,
        measurement_noise_sqrt,
        MerweScaled { alpha, beta, kappa },
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
        ukf.predict(dt, None);
        let state = ukf.get_state().0;
        let cov_sqrt = ukf.get_sqrt_covariance();
        let cov = cov_sqrt * cov_sqrt.transpose();
        let pos_std_dev = cov[(0, 0)].sqrt();
        let vel_std_dev = cov[(1, 1)].sqrt();

        // Log predicted state
        rec.log("predicted/position", &Scalars::new([state[0]]))?;
        rec.log("predicted/velocity", &Scalars::new([state[1]]))?;
        rec.log("predicted/position_std_dev", &Scalars::new([pos_std_dev]))?;
        rec.log("predicted/velocity_std_dev", &Scalars::new([vel_std_dev]))?;

        // Update
        ukf.update(measurement);
        let state_updated = ukf.get_state().0;
        let cov_sqrt_updated = ukf.get_sqrt_covariance();
        let cov_updated = cov_sqrt_updated * cov_sqrt_updated.transpose();
        let pos_std_dev_updated = cov_updated[(0, 0)].sqrt();
        let vel_std_dev_updated = cov_updated[(1, 1)].sqrt();

        // Log updated state
        rec.log("updated/position", &Scalars::new([state_updated[0]]))?;
        rec.log("updated/velocity", &Scalars::new([state_updated[1]]))?;
        rec.log(
            "updated/position_std_dev",
            &Scalars::new([pos_std_dev_updated]),
        )?;
        rec.log(
            "updated/velocity_std_dev",
            &Scalars::new([vel_std_dev_updated]),
        )?;

        // Log errors
        let pos_error = state_updated[0] - true_pos;
        let vel_error = state_updated[1] - true_vel;
        rec.log("error/position", &Scalars::new([pos_error]))?;
        rec.log("error/velocity", &Scalars::new([vel_error]))?;

        // Log covariance matrix
        let cov_array = Array::from_shape_vec((2, 2), cov_updated.transpose().as_slice().to_vec())?;
        let tensor = Tensor::try_from(cov_array)?.with_dim_names(["rows", "cols"]);
        rec.log("updated/covariance", &tensor)?;
    }

    Ok(())
}
