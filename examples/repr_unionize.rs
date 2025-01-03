use estimacros::repr_unionize;

repr_unionize! {
    struct Speed { x: f32, y: f32, z: f32 }
}

repr_unionize! {
    struct State { x: f32, y: f32, z: f32, v: Speed }
}

fn main() {
    let v = Speed {
        values: [1.0, 2.0, 3.0],
    };

    let mut state = State {
        fields: StateFields {
            x: 4.0,
            y: 5.0,
            z: 6.0,
            v,
        },
    };

    // Using field names
    println!("> state.x: {}, state[0]: {}", state.x, state[0]);
    state.x = 7.0;
    println!("< state.x: {}, state[0]: {}", state.x, state[0]);

    // Using index
    println!("> state.x: {}, state[0]: {}", state.x, state[0]);
    state[0] = 8.0;
    println!("< state.x: {}, state[0]: {}", state.x, state[0]);

    // Access nested fields
    println!("> state.v.y: {}, state.v[1]: {}", state.v.y, state.v[1]);
    state.v.y = 0.0;
    println!("< state.v.y: {}, state.v[1]: {}", state.v.y, state.v[1]);
}
