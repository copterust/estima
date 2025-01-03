use estimacros::repr_unionize;

repr_unionize! {
    struct Speed { x: f32, y: f32, z: f32 }
}

fn main() {
    let mut speed = Speed {
        fields: SpeedFields {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
    };

    // Modify using field names
    speed.x = 3.0;
    println!("speed.x: {}, speed[0]: {}", speed.x, speed[0]);

    // Access and modify using index
    speed[0] = 2.0;
    println!("speed.x: {}, speed[0]: {}", speed.x, speed[0]);
}
