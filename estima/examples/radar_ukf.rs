//! # Radar Tracking with Unscented Kalman Filter (UKF) on Manifolds
//!
//! This example demonstrates how to use the Unscented Kalman Filter (UKF) with
//! **Composite Manifolds** to solve a tracking problem in a "Finger Pointing" scenario.
//!
//! ## The Scenario: "Finger Pointing"
//! Imagine you are pointing your finger at a flying plane.
//! - Your finger direction (a point on the sphere $S^2$) and the distance to the plane
//!   define its position.
//! - The plane flies with a constant velocity in 3D Cartesian space.
//!
//! We want to track the plane's state using this mixed representation:
//! - **Position**: Represented as (Direction $\in S^2$, Range $\in \mathbb{R}^+$).
//! - **Velocity**: Represented in standard Cartesian coordinates ($\in \mathbb{R}^3$).
//!
//! ## The Manifold Structure
//! The state space is a **Composite Manifold**:
//!
//! $$ \text{State} = (\underbrace{S^2 \times \mathbb{R}^1}_{\text{Position (Spherical)}}) \times \underbrace{\mathbb{R}^3}_{\text{Velocity}} $$
//!
//! - **$S^2$ (Sphere)**: Handles the direction. It naturally handles the topology of angles,
//!   avoiding singularities and wrap-around issues (gimbal lock) inherent in Euler angles.
//! - **$\mathbb{R}^1$ (Euclidean)**: Handles the scalar range.
//! - **$\mathbb{R}^3$ (Euclidean)**: Handles the velocity vector.
//!
//! ## Why is this cool?
//! 1. **No Angle Wrapping**: The UKF generates sigma points directly on the sphere. We don't
//!    need to worry about $359^\circ$ vs $1^\circ$ or singularities at the poles.
//! 2. **Geometric Consistency**: The filter respects the geometry of the problem.
//! 3. **Composite Flexibility**: We mix different types of manifolds (Spherical + Euclidean)
//!    seamlessly.
//!
//! ## The Models
//! - **Process Model**: Constant Velocity with Manifold Update.
//!     1. Decompose Velocity into **Radial** and **Tangential** components.
//!     2. Update Range using Radial velocity.
//!     3. Update Direction using `retract` with Tangential velocity (converted to angular rate).
//!     - This avoids converting the full state to Cartesian coordinates, respecting the manifold structure.
//! - **Measurement Model**: We observe Direction and Range directly.
//!
//! ## Visualization
//! If you have `rerun` installed, this example will log the true trajectory, measurements,
//! and filter estimates for visualization.

use estima::manifold::{
    composite::CompositeManifold,
    euclidean::EuclideanManifold,
    s2::S2Manifold,
    CompositeStrategy, EuclideanMean, Manifold, ManifoldMeasurement, ManifoldProcess,
};
use estima::sigma_points::MerweScaledSigmaPoints;
use estima::UnscentedKalmanFilter;
use nalgebra::{
    OMatrix, OVector, Unit, Vector3, Vector4, U1, U2, U3, U4, U6,
};
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::error::Error;

// -----------------------------------------------------------------------------
// 1. Define the State
// -----------------------------------------------------------------------------

// Part 1: Position = Direction (S^2) + Range (R^1)
// We use a CompositeManifold to combine them.
// Tangent Space Dim: 2 (for S^2) + 1 (for R^1) = 3
type SphericalPos = CompositeManifold<f64, S2Manifold<f64>, EuclideanManifold<f64, U1>, U2, U1>;

// Part 2: Velocity = R^3
type Velocity = EuclideanManifold<f64, U3>;

// Full State = Position + Velocity
// Tangent Space Dim: 3 (Pos) + 3 (Vel) = 6
// Measurement Space Dim: 3 (Pos Tangent) + 3 (Vel Tangent) = 6
type RadarState = CompositeManifold<f64, SphericalPos, Velocity, U3, U3>;

/// Helper to create a state from direction, range, and velocity
fn create_state(dir: Unit<Vector3<f64>>, range: f64, vel: Vector3<f64>) -> RadarState {
    let pos = SphericalPos::new(
        S2Manifold::new(dir),
        EuclideanManifold::new(OVector::<f64, U1>::new(range)),
    );
    RadarState::new(pos, Velocity::new(vel))
}

/// Helper to extract Cartesian position from the state
fn get_cartesian_pos(state: &RadarState) -> Vector3<f64> {
    let dir = state.first.first.as_unit_vector();
    let range = state.first.second.as_vector()[0];
    dir.as_ref() * range
}

/// Helper to extract velocity from the state
fn get_velocity(state: &RadarState) -> Vector3<f64> {
    *state.second.as_vector()
}

// -----------------------------------------------------------------------------
// 2. Define the Process Model (The Physics)
// -----------------------------------------------------------------------------
// We assume a "Constant Velocity" model.
//
// Instead of converting to Cartesian coordinates, we perform the update directly
// on the manifold components by decomposing the velocity vector.
//
// 1. Radial Velocity ($v_r$): Updates the Range.
// 2. Tangential Velocity ($v_t$): Updates the Direction via retraction.
#[derive(Clone)]
struct ConstantVelocityModel;

impl ManifoldProcess<RadarState, U3, f64> for ConstantVelocityModel {
    fn predict(
        &self,
        state: &RadarState,
        dt: f64,
        _control: Option<&OVector<f64, U3>>,
    ) -> RadarState {
        // 1. Extract components
        let dir_manifold = &state.first.first;
        let dir = dir_manifold.as_unit_vector();
        let range = state.first.second.as_vector()[0];
        let vel = get_velocity(state);

        // 2. Decompose Velocity
        // Radial velocity = V . dir
        let v_radial = vel.dot(dir.as_ref());

        // Tangential velocity vector (in 3D)
        // V_tan = V - (V . dir) * dir
        let v_tan_3d = vel - dir.as_ref() * v_radial;

        // 3. Update Range
        // r_new = r + v_r * dt
        let new_range = range + v_radial * dt;

        // 4. Update Direction (Manifold Retract)
        // Angular displacement magnitude = |V_tan| * dt / range
        // We need to project v_tan_3d into the 2D tangent space of the S2 manifold.
        // We need a local basis (e1, e2) at 'dir'.
        let (e1, e2) = local_basis(dir.as_ref());

        // Project 3D tangential velocity onto 2D basis
        // The tangent vector delta is in radians.
        // Arc length = r * theta => theta = arc_length / r
        // Here arc_length approx |v_tan| * dt
        let omega_1 = v_tan_3d.dot(&e1) / range;
        let omega_2 = v_tan_3d.dot(&e2) / range;

        let delta_theta = nalgebra::Vector2::new(omega_1, omega_2) * dt;
        let new_dir_manifold = dir_manifold.retract(&delta_theta);

        // 5. Construct new state
        // Note: Velocity is constant in Cartesian frame, so we keep it as is.
        // (In a true curved space model, we'd need parallel transport, but here
        // the velocity space is global Euclidean R3).
        let new_pos = SphericalPos::new(
            new_dir_manifold,
            EuclideanManifold::new(OVector::<f64, U1>::new(new_range)),
        );
        RadarState::new(new_pos, Velocity::new(vel))
    }
}

/// Helper to construct a local basis on S2 (adapted from estima::manifold::s2)
fn local_basis(p: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let threshold = 0.9;
    let a = if p.x.abs() > threshold {
        Vector3::y()
    } else {
        Vector3::x()
    };
    let e1 = p.cross(&a).normalize();
    let e2 = p.cross(&e1);
    (e1, e2)
}

// -----------------------------------------------------------------------------
// 3. Define the Measurement Model (The Sensor)
// -----------------------------------------------------------------------------
// Our sensor measures the "Finger Pointing" parameters directly:
// - Direction (Unit Vector in R^3)
// - Range (Scalar)
//
// Note: We model the measurement as a vector in R^4: [dx, dy, dz, range].
// Ideally, we could treat the measurement space as a manifold too (S^2 x R^1),
// but `ManifoldMeasurement` typically produces a vector in the measurement space.
//
// Here, we'll output the measurement as a 4D vector: [nx, ny, nz, range].
// The residual function will handle the geometry.

#[derive(Clone)]
struct RadarMeasurementModel;

impl ManifoldMeasurement<RadarState, U6, U4, f64> for RadarMeasurementModel {
    fn measure(&self, state: &RadarState) -> OVector<f64, U4> {
        let dir = state.first.first.as_unit_vector();
        let range = state.first.second.as_vector()[0];

        Vector4::new(dir.x, dir.y, dir.z, range)
    }

    /// The residual function computes the difference between predicted and actual measurements.
    ///
    /// - For Range: Simple scalar difference.
    /// - For Direction: We need a vector in the tangent space of the measurement.
    ///   Since we are outputting a 3D direction vector, the error is not just (z - h(x)).
    ///   However, for simplicity in this example (and since standard UKF expects vector residuals),
    ///   we will treat the direction error as the vector difference in R^3.
    ///   Strictly speaking, a more "manifold-correct" way would be to return the error in the
    ///   tangent space of S^2 (2D), but that would require changing the Measurement dimension to U3.
    ///
    ///   Let's stick to U4 measurement [nx, ny, nz, r] and simple subtraction.
    ///   The UKF will project this error into the state space.
    ///   Ideally, we would use an "Innovation" that respects S^2, but simple vector subtraction
    ///   on unit vectors works reasonably well for small errors.
    fn residual(
        &self,
        predicted: &OVector<f64, U4>,
        measured: &OVector<f64, U4>,
    ) -> OVector<f64, U4> {
        measured - predicted
    }
}

// -----------------------------------------------------------------------------
// 4. Visualization (Optional)
// -----------------------------------------------------------------------------
#[cfg(feature = "rerun")]
struct Visualizer {
    rec: rerun::RecordingStream,
}

#[cfg(feature = "rerun")]
impl Visualizer {
    fn new() -> Result<Self, Box<dyn Error>> {
        let rec = rerun::RecordingStreamBuilder::new("estima_radar_ukf").spawn()?;
        Ok(Self { rec })
    }

    fn log_sigma_points(
        &self,
        sigmas: &[RadarState],
    ) -> Result<(), Box<dyn Error>> {
        use rerun::{archetypes::Points3D, Color};

        let points: Vec<_> = sigmas
            .iter()
            .map(|s| {
                let pos = get_cartesian_pos(s);
                (pos.x as f32, pos.y as f32, pos.z as f32)
            })
            .collect();

        self.rec.log(
            "world/sigma_cloud",
            &Points3D::new(points)
                .with_colors([Color::from_rgb(200, 200, 200)])
                .with_radii([0.2]),
        )?;
        Ok(())
    }

    fn log_ray(
        &self,
        origin: Vector3<f64>,
        dir: Vector3<f64>,
        length: f64,
        entity_path: &str,
        color: [u8; 3],
    ) -> Result<(), Box<dyn Error>> {
        use rerun::{archetypes::LineStrips3D, Color};

        let end = origin + dir * length;
        let points = vec![
            (origin.x as f32, origin.y as f32, origin.z as f32),
            (end.x as f32, end.y as f32, end.z as f32),
        ];

        self.rec.log(
            entity_path,
            &LineStrips3D::new([points])
                .with_colors([Color::from_rgb(color[0], color[1], color[2])])
                .with_radii([0.1]),
        )?;
        Ok(())
    }

    fn log_step(
        &self,
        step: i64,
        true_pos: Vector3<f64>,
        est_pos: Vector3<f64>,
        meas_pos: Vector3<f64>,
        cov: &OMatrix<f64, U6, U6>,
    ) -> Result<(), Box<dyn Error>> {
        use rerun::{
            archetypes::{Points3D, Scalars},
            Color,
        };

        self.rec.set_time_sequence("step", step);

        // 1. Log True Position (Green)
        self.rec.log(
            "world/truth",
            &Points3D::new([(true_pos.x as f32, true_pos.y as f32, true_pos.z as f32)])
                .with_colors([Color::from_rgb(60, 179, 113)])
                .with_radii([0.5]),
        )?;

        // 2. Log Estimated Position (Blue)
        self.rec.log(
            "world/estimate",
            &Points3D::new([(est_pos.x as f32, est_pos.y as f32, est_pos.z as f32)])
                .with_colors([Color::from_rgb(30, 144, 255)])
                .with_radii([0.5]),
        )?;

        // 3. Log Measurement (Orange)
        self.rec.log(
            "world/measurement",
            &Points3D::new([(meas_pos.x as f32, meas_pos.y as f32, meas_pos.z as f32)])
                .with_colors([Color::from_rgb(255, 140, 0)])
                .with_radii([0.3]),
        )?;

        // 4. Log Error Metrics
        let pos_error = (true_pos - est_pos).norm();
        self.rec.log("errors/position", &Scalars::new([pos_error]))?;

        // 5. Log Uncertainty (Trace of Position Covariance)
        // The covariance is 6x6. The first 3x3 block corresponds to the composite position manifold.
        // Note: The covariance is in the TANGENT space.
        // Tangent indices: 0,1 (S^2), 2 (Range), 3,4,5 (Velocity).
        // It's hard to map directly to Cartesian variance without Jacobian, but we can log the trace.
        let pos_tangent_trace = cov[(0, 0)] + cov[(1, 1)] + cov[(2, 2)];
        self.rec
            .log("uncertainty/pos_tangent_trace", &Scalars::new([pos_tangent_trace]))?;

        Ok(())
    }
}

// -----------------------------------------------------------------------------
// 5. Main Simulation Loop
// -----------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "rerun")]
    let viz = Visualizer::new()?;

    // --- Simulation Parameters ---
    let dt = 0.1; // Time step (seconds)
    let total_steps = 300;

    // Noise parameters
    // Process noise is applied in the tangent space of the state.
    // Tangent dims: [d_theta1, d_theta2, d_range, d_vx, d_vy, d_vz]
    let process_noise_dir: f64 = 0.001;
    let process_noise_range: f64 = 0.1;
    let process_noise_vel: f64 = 0.1;

    // Measurement noise
    // [n_x, n_y, n_z, range]
    let meas_noise_dir_vec: f64 = 0.05; // Noise on direction vector components
    let meas_noise_range: f64 = 5.0;

    // --- Initial State ---
    // Object starts at [100, 0, 50], moving in Y direction
    let mut true_cartesian_pos = Vector3::new(100.0, 0.0, 50.0);
    let true_velocity = Vector3::new(0.0, 20.0, 5.0);

    // Filter Initial Guess
    // Slightly off
    let initial_pos_guess = Vector3::new(110.0, -10.0, 45.0);
    let initial_vel_guess = Vector3::new(0.0, 18.0, 4.0);

    let initial_state = create_state(
        Unit::new_normalize(initial_pos_guess),
        initial_pos_guess.norm(),
        initial_vel_guess,
    );

    // Initial Covariance (Tangent Space 6x6)
    let mut initial_cov = OMatrix::<f64, U6, U6>::identity();
    initial_cov.fill_diagonal(1.0);

    // --- Setup UKF ---
    // 1. Process Noise Matrix (Q) - 6x6
    let mut q = OMatrix::<f64, U6, U6>::zeros();
    q[(0, 0)] = process_noise_dir.powi(2);
    q[(1, 1)] = process_noise_dir.powi(2);
    q[(2, 2)] = process_noise_range.powi(2);
    q[(3, 3)] = process_noise_vel.powi(2);
    q[(4, 4)] = process_noise_vel.powi(2);
    q[(5, 5)] = process_noise_vel.powi(2);

    // 2. Measurement Noise Matrix (R) - 4x4
    let mut r = OMatrix::<f64, U4, U4>::zeros();
    r[(0, 0)] = meas_noise_dir_vec.powi(2);
    r[(1, 1)] = meas_noise_dir_vec.powi(2);
    r[(2, 2)] = meas_noise_dir_vec.powi(2);
    r[(3, 3)] = meas_noise_range.powi(2);

    // 3. Sigma Points Generator
    let sigma_points = MerweScaledSigmaPoints::new(0.1, 2.0, 0.0);
    let weights = sigma_points.weights::<U6>();

    // 4. Manifold Averaging Strategy
    use estima::manifold::FrechetMean;
    
    let pos_strategy = CompositeStrategy::new(
        FrechetMean { tolerance: 1e-5, max_iterations: 10 }, // For S2
        EuclideanMean,              // For Range
    );
    let full_strategy = CompositeStrategy::new(
        pos_strategy,
        EuclideanMean, // For Velocity
    );

    // 5. Build the Filter
    let mut ukf = UnscentedKalmanFilter::new(
        initial_state,
        initial_cov,
        ConstantVelocityModel,
        q,
        RadarMeasurementModel,
        r,
        sigma_points,
        weights,
        full_strategy,
    );

    let mut rng = StdRng::seed_from_u64(42);

    println!(
        "{:<5} | {:<25} | {:<25} | {:<10}",
        "Step", "True Pos (x,y,z)", "Est Pos (x,y,z)", "Error"
    );
    println!("{:-<75}", "");

    for step in 0..total_steps {
        // 1. Move the true object (Physics Simulation)
        true_cartesian_pos += true_velocity * dt;

        // 2. Generate Noisy Measurement (Sensor Simulation)
        let true_range = true_cartesian_pos.norm();
        let true_dir = Unit::new_normalize(true_cartesian_pos);

        let meas_range = true_range + rng.sample::<f64, _>(StandardNormal) * meas_noise_range;
        
        // Add noise to direction vector
        let noise_vec = Vector3::new(
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
        );
        // Project noise onto tangent plane to be somewhat realistic, or just add and renormalize
        let noisy_dir_vec = true_dir.as_ref() + noise_vec * meas_noise_dir_vec;
        let meas_dir = Unit::new_normalize(noisy_dir_vec);

        let measurement = Vector4::new(meas_dir.x, meas_dir.y, meas_dir.z, meas_range);

        // 3. UKF Prediction Step
        ukf.predict(dt, None)?;

        // 4. UKF Update Step
        ukf.update(&measurement)?;

        // --- Logging & Visualization ---
        let estimate_state = ukf.nominal_state();
        let est_pos = get_cartesian_pos(estimate_state);
        
        let pos_error = (true_cartesian_pos - est_pos).norm();

        if step % 20 == 0 {
            println!(
                "{:<5} | ({:6.1}, {:6.1}, {:6.1}) | ({:6.1}, {:6.1}, {:6.1}) | {:6.2}m",
                step,
                true_cartesian_pos.x, true_cartesian_pos.y, true_cartesian_pos.z,
                est_pos.x, est_pos.y, est_pos.z,
                pos_error
            );
        }

        #[cfg(feature = "rerun")]
        {
            let meas_pos = meas_dir.as_ref() * meas_range;
            let (_, cov) = ukf.state_with_covariance();
            viz.log_step(
                step as i64,
                true_cartesian_pos,
                est_pos,
                meas_pos,
                &cov,
            )?;

            // Log Sigma Points (Cloud)
            viz.log_sigma_points(ukf.sigma_points())?;

            // Log Measurement Ray (Ping)
            viz.log_ray(
                Vector3::zeros(), // Origin
                meas_dir.into_inner(),
                meas_range,
                "world/measurement_ray",
                [255, 140, 0], // Orange
            )?;
        }
    }

    println!("\nSimulation Complete.");
    println!("\n👉 Manifold Note:");
    println!("The state uses a Composite Manifold: S^2 (Direction) x R^1 (Range) x R^3 (Velocity).");
    println!("The UKF sigma points for direction are generated on the sphere surface,");
    println!("ensuring geometric consistency without Euler angle singularities!");
    Ok(())
}
