//! CVM (constant velocity model) example using estima-rs
use estima::{MeasurementModel, ProcessModel, State, UKFBuilder};

/// State vector, [position_x, position_y, velocity_x, velocity_y]
#[derive(Debug, Copy, Clone)]
struct ConstantVelocityState {
    data: [f32; 4],
}

impl ConstantVelocityState {
    fn new(position: [f32; 2], velocity: [f32; 2]) -> Self {
        let mut data = [0.0; 4];
        data[0..2].copy_from_slice(&position);
        data[2..4].copy_from_slice(&velocity);

        Self { data }
    }
}

impl State<4> for ConstantVelocityState {
    type Scalar = f32;

    fn as_array(&self) -> &[Self::Scalar; 4] {
        &self.data
    }

    fn from_array(data: &[Self::Scalar; 4]) -> Self {
        Self { data: *data }
    }
}

struct ConstantVelocityDynamics;

impl ProcessModel<ConstantVelocityState, 4> for ConstantVelocityDynamics {
    fn predict(
        &self,
        state: &ConstantVelocityState,
        dt: f32,
        _control: Option<&[f32]>,
    ) -> ConstantVelocityState {
        let new_position = [
            state.data[0] + state.data[2] * dt,
            state.data[1] + state.data[3] * dt,
        ];
        let mut data = [0.0; 4];
        data[0..2].copy_from_slice(&new_position);
        data[2..4].copy_from_slice(&state.data[2..4]); // Velocity is constant

        ConstantVelocityState { data }
    }
}

struct PositionMeasurement;

impl MeasurementModel<ConstantVelocityState, [f32; 2], 4> for PositionMeasurement {
    fn measure(&self, state: &ConstantVelocityState) -> [f32; 2] {
        [state.data[0], state.data[1]]
    }

    fn residual(&self, predicted: &[f32; 2], measurement: &[f32; 2]) -> [f32; 2] {
        [measurement[0] - predicted[0], measurement[1] - predicted[1]]
    }
}

static MEASUREMENTS: [[f32; 2]; 5] = [[1.0, 2.0], [1.1, 2.1], [1.4, 2.5], [1.8, 3.0], [2.0, 3.4]];

fn log_state(step: usize, state: &ConstantVelocityState) {
    println!("Step {}, state: {:?}", step, state);
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
        ukf.update(&measurement);
        let state = ukf.state();
        log_state(step, state);
    }
}
