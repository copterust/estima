//! CVM (constant velocity model) example using estima-rs
use estima::{ProcessModel, MeasurementModel, UKFBuilder};

/// State vector, [position_x, position_y, velocity_x, velocity_y]
#[derive(Debug, Copy, Clone)]
struct ConstantVelocityState {
    position: [f32; 2],
    velocity: [f32; 2],
}

impl ConstantVelocityState {
    fn new(position: [f32; 2], velocity: [f32; 2]) -> Self {
        Self { position, velocity }
    }
}

struct ConstantVelocityDynamics;

impl ProcessModel<ConstantVelocityState> for ConstantVelocityDynamics {
    fn predict(
        &self,
        state: &ConstantVelocityState,
        dt: f32,
        _control: Option<&[f32]>,
    ) -> ConstantVelocityState {
        let new_position = [
            state.position[0] + state.velocity[0] * dt,
            state.position[1] + state.velocity[1] * dt,
        ];
        ConstantVelocityState {
            position: new_position,
            velocity: state.velocity, // Velocity is constant
        }
    }
}

struct PositionMeasurement;

impl MeasurementModel<ConstantVelocityState, [f32; 2]> for PositionMeasurement {
    fn measure(&self, state: &ConstantVelocityState) -> [f32; 2] {
        state.position
    }

    fn residual(&self, predicted: [f32; 2], measurement: [f32; 2]) -> [f32; 2] {
        [measurement[0] - predicted[0], measurement[1] - predicted[1]]
    }
}

static MEASUREMENTS: [[f32; 2]; 5] = [[1.0, 2.0], [1.1, 2.1], [1.4, 2.5], [1.8, 3.0], [2.0, 3.4]];

fn log_state(step: usize, state: &ConstantVelocityState) {
    println!(
        "Step {}: Position: {:?}, Velocity: {:?}",
        step, state.position, state.velocity
    );
}

fn main() {
    let initial_state = ConstantVelocityState::new([0.0, 0.0], [1.0, 1.0]);

    let mut ukf = UKFBuilder::new()
        .with_state(initial_state)
        .with_process(ConstantVelocityDynamics)
        .with_measurement(PositionMeasurement)
        .build();

    for (step, &measurement) in MEASUREMENTS.iter().enumerate() {
        ukf.predict(1.0);
        ukf.update(measurement);
        let state = ukf.state();
        log_state(step, state);
    }
}
