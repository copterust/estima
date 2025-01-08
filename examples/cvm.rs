//! CVM (constant velocity model) example using estima-rs
use estima::{MeasurementModel, ProcessModel, State, UKFBuilder};
use estimacros::vector_union;

vector_union! { Position, f32, PositionFields { x, y } }
vector_union! { Velocity, f32, VelocityFields { x, y } }
vector_union! { ConstantVelocityState, f32, StateFields { position: Position, v: Velocity } }

impl ConstantVelocityState {
    fn new(position: Position, velocity: Velocity) -> Self {
        Self {
            fields: StateFields {
                position,
                v: velocity,
            },
        }
    }
}

impl State<4> for ConstantVelocityState {
    type Scalar = f32;

    fn as_array(&self) -> &[Self::Scalar; 4] {
        unsafe { &self.values }
    }

    fn from_array(data: &[Self::Scalar; 4]) -> Self {
        Self { values: *data }
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
        let new_position = Position {
            fields: PositionFields {
                x: state.position.x + state.v.x * dt,
                y: state.position.y + state.v.y * dt,
            },
        };

        ConstantVelocityState {
            fields: StateFields {
                position: new_position,
                v: state.v, // Velocity is constant
            },
        }
    }
}

struct PositionMeasurement;

impl MeasurementModel<ConstantVelocityState, Position, 4> for PositionMeasurement {
    fn measure(&self, state: &ConstantVelocityState) -> Position {
        state.position
    }

    fn residual(&self, predicted: &Position, measurement: &Position) -> Position {
        Position {
            fields: PositionFields {
                x: measurement.x - predicted.x,
                y: measurement.y - predicted.y,
            },
        }
    }
}

static MEASUREMENTS: [[f32; 2]; 5] = [[1.0, 2.0], [1.1, 2.1], [1.4, 2.5], [1.8, 3.0], [2.0, 3.4]];

fn log_state(step: usize, state: &ConstantVelocityState) {
    println!(
        "Step {}, x: {}, y: {}, v_x: {}, v_y: {}",
        step, state.position.x, state.position.y, state.v.x, state.v.y
    );
}

fn main() {
    let initial_position_guess = Position { values: [0.0, 0.0] };
    let initial_velocity_guess = Velocity { values: [1.0, 2.0] };
    let initial_state = ConstantVelocityState::new(initial_position_guess, initial_velocity_guess);

    let mut ukf = UKFBuilder::new()
        .with_state(initial_state)
        .with_process(ConstantVelocityDynamics)
        .with_measurement(PositionMeasurement)
        .build();

    for (step, &measurement) in MEASUREMENTS.iter().enumerate() {
        ukf.predict(1.0);
        let measured_position = Position {
            values: measurement,
        };
        ukf.update(&measured_position);
        let state = ukf.state();
        log_state(step, state);
    }
}
