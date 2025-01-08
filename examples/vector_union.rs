use estimacros::vector_union;

vector_union! { Point, f32, PointFields { x, y, z } }
vector_union! { State, f32, StateFields { w, p: Point, f, d: Point } }

fn main() {
    let p = Point {
        values: [2.0, 3.0, 4.0],
    };

    let d = Point {
        values: [6.0, 7.0, 8.0],
    };

    let mut state = State {
        fields: StateFields {
            w: 1.0,
            p,
            f: 5.0,
            d,
        },
    };

    println!("state.d.z: {}, state[7]: {}", state.d.z, state[7]);
    state.d.z = -1.0;
    println!("state.d.z: {}, state[7]: {}", state.d.z, state[7]);
    state[7] = -2.0;
    println!("state.d.z: {}, state[7]: {}", state.d.z, state[7]);
}
