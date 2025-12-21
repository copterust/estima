use estimacros::define_ukf_system;

define_ukf_system! {
    type Float = f64;

    state CVState {
        pos: f64,
        vel: f64,
    }

    process CVProcessModel(dt: f64) {
        next.pos = state.pos + state.vel * dt;
        next.vel = state.vel;
    }
}

fn main() {
    let state = CVState::new(0.0, 1.0);
    println!("Initial state -> pos: {}, vel: {}", state.pos(), state.vel());

    let process_model = CVProcessModel;
    let dt = 0.1;
    let next_state = process_model.predict(&state, dt, None);

    println!("Predicted state -> pos: {}, vel: {}", next_state.pos(), next_state.vel());
}