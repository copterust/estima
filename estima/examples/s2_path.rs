//! S^2 Manifold Robot Arm Example
//!
//! This example provides a simple, intuitive demonstration of why the S^2 manifold
//! is important for working with directions in 3D space, such as in robotics.
//!
//! Imagine arrow on a ball. The direction it points is a unit vector, a point on the S^2 sphere.
//!
//! We want to move the arm smoothly from a `start_direction` to a `target_direction`.
//!
//! # The Problem with a Naive Approach
//! A simple way to interpolate between two 3D vectors is linear interpolation (lerp),
//! followed by re-normalizing the result to keep it on the unit sphere.
//!
//! `interpolated = (1 - t) * start + t * target`
//! `final_direction = normalize(interpolated)`
//!
//! This path does *not* trace the shortest path across the sphere's surface (a
//! great-circle arc). This means the arm moves inefficiently and, more importantly,
//! its angular velocity changes throughout the movement, which is undesirable in robotics.
//!
//! # The Manifold Solution
//! The S^2 manifold provides operations that respect the geometry of the sphere.
//!
//! 1. `local(start, target)`: This calculates the tangent vector representing the
//!    shortest "straight line" path from start to target *on the sphere's surface*.
//! 2. `retract(start, tangent * t)`: This moves the `start` point along the geodesic
//!    (the great-circle arc) defined by the tangent vector.
//!
//! This results in the most efficient movement with a constant angular velocity.

use estima::manifold::s2::S2Manifold;
use estima::manifold::Manifold;
use nalgebra::{Unit, Vector3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    type T = f64;

    let start_vec_euclid = Vector3::new(1.0, 0.0, 0.0);
    let target_vec_euclid = Vector3::new(1.0, 1.0, 1.0);

    let start_point = S2Manifold::<T>::new(Unit::new_normalize(start_vec_euclid));
    let target_point = S2Manifold::<T>::new(Unit::new_normalize(target_vec_euclid));

    let steps = 5;
    let normalized_target = target_point.as_unit_vector().into_inner();
    let normalized_start = start_point.as_unit_vector().into_inner();
    println!(
        "Start  {:6.2?} | euclid {:6.2?}",
        normalized_start, start_vec_euclid
    );
    println!(
        "Target {:6.2?} | euclid {:6.2?}\n",
        normalized_target, target_vec_euclid
    );

    // 1. The Manifold Approach (Correct Way)
    println!("Manifold Interpolation (constant velocity):");
    // `local` gives us the tangent vector for the shortest path on the sphere.
    let tangent_delta = start_point.local(&target_point);
    let mut manifold_prev = start_point.as_unit_vector().into_inner();

    for i in 0..=steps {
        let t = i as T / steps as T;
        // `retract` moves the start point along the great-circle arc.
        let manifold_interpolated = start_point.retract(&(tangent_delta * t));
        let point_vec = manifold_interpolated.as_unit_vector().into_inner();

        let angle_step_rad = if i > 0 {
            manifold_prev.angle(&point_vec)
        } else {
            0.0
        };

        println!(
            "  t={:.1}: ({:6.2}, {:6.2}, {:6.2}) | Angle step: {:.2}°",
            t,
            point_vec.x,
            point_vec.y,
            point_vec.z,
            angle_step_rad.to_degrees()
        );

        manifold_prev = point_vec;
    }

    // 2. The Naive Approach (Incorrect Way)
    println!("\nNaive Linear Interpolation (variable velocity):");
    let mut naive_prev = start_vec_euclid;

    for i in 0..=steps {
        let t = i as T / steps as T;
        // Linearly interpolate in 3D Euclidean space...
        let naively_interpolated = start_vec_euclid.lerp(&target_vec_euclid, t);
        // ...and then project it back onto the sphere by normalizing.
        let point_vec = naively_interpolated.normalize();

        let angle_step_rad = if i > 0 {
            naive_prev.angle(&point_vec)
        } else {
            0.0
        };

        println!(
            "  t={:.1}: ({:6.2}, {:6.2}, {:6.2}) | Angle step: {:.2}°",
            t,
            point_vec.x,
            point_vec.y,
            point_vec.z,
            angle_step_rad.to_degrees()
        );

        naive_prev = point_vec;
    }

    println!(
        "\nNotice the 'Angle step' column. The manifold approach has a constant angular velocity."
    );

    Ok(())
}
