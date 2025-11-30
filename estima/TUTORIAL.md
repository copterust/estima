# Estima: The Manifold Kalman Filter Tutorial

![Radar UKF Concept](/Users/r/.gemini/antigravity/brain/1b70410d-8b0d-405f-a70b-87df539e9de1/radar_ukf_concept_1764524987388.png)

Welcome to **Estima**, a Rust library for state estimation that takes geometry seriously.

If you've ever struggled with:
- **Gimbal lock** when tracking orientation.
- **Angle wrapping** bugs (e.g., $359^\circ \approx 1^\circ$).
- **Normalization issues** where your quaternions drift away from unit length.

Then you need **Manifolds**.

---

## 1. The Intuition: The Flat Earth Problem

Standard Kalman Filters (EKF/UKF) assume the world is a **Vector Space**. They assume:
1. You can add any two states: $x_1 + x_2$.
2. You can multiply any state by a scalar: $2 \cdot x$.
3. Straight lines go on forever.

**This is the "Flat Earth" assumption.**

But the real world is curved.
- **Directions** live on a sphere ($S^2$). If you keep walking north, you don't go to space; you wrap around.
- **Orientations** live on a hypersphere ($S^3$ / Unit Quaternions).
- **Poses** live on the Special Euclidean Group ($SE(3)$).

### The "Hack" vs The Solution
- **The Hack**: Use Euler angles. Result: Singularities at poles (Gimbal lock).
- **The Hack**: Use Quaternions but treat them as 4D vectors. Result: They stop being unit length, math breaks.
- **The Solution (Manifolds)**: Treat the state as living on a curved surface. Do the math (averaging, uncertainty) on a flat **Tangent Space** that touches the surface at the current estimate.

---

## 2. Core Concepts

To use Estima, you only need to understand three operations:

### 1. The Manifold ($M$)
The curved space where your state lives.
- Example: The surface of a sphere.

### 2. The Tangent Space ($T_x M$)
A flat Euclidean space tangent to the manifold at a point $x$.
- Example: A 2D plane touching the sphere at the North Pole.
- **Key Idea**: We define uncertainty (covariance) and "deltas" here.

### 3. Retract ($\oplus$)
How we move from the flat tangent space back to the manifold.
- "Walk this direction on the plane, then wrap it onto the sphere."
- $x_{new} = x \oplus \delta$

### 4. Local / Log ($\ominus$)
How we measure the difference between two points on the manifold as a tangent vector.
- "What straight line on the map corresponds to the path between A and B?"
- $\delta = y \ominus x$

---

## 3. Estima Architecture

Estima provides traits to handle this automatically.

### The `Manifold` Trait
```rust
pub trait Manifold<TangentDim, T>: Clone {
    // The "Plus" operator
    fn retract(&self, delta: &OVector<T, TangentDim>) -> Self;

    // The "Minus" operator
    fn local(&self, other: &Self) -> OVector<T, TangentDim>;
}
```

### Built-in Manifolds
- `EuclideanManifold<N>`: Standard vectors ($\mathbb{R}^N$). Retract is just addition.
- `S2Manifold`: Directions on a sphere. Tangent space is 2D.
- `UnitQuaternionManifold`: Rotations. Tangent space is 3D (rotation vector).
- `CompositeManifold`: Combine them! (e.g., Pose = Rotation + Position).

---

## 4. Hands-On: The "Finger Pointing" Radar

Let's build a tracker for a "Finger Pointing" scenario.
- **State**:
    - **Direction**: Where you are pointing ($S^2$).
    - **Range**: How far away it is ($\mathbb{R}^1$).
    - **Velocity**: Moving in 3D space ($\mathbb{R}^3$).

This is a **Composite Manifold**: $(S^2 \times \mathbb{R}^1) \times \mathbb{R}^3$.

### Step 1: Define the State
```rust
use estima::manifold::{
    composite::CompositeManifold,
    euclidean::EuclideanManifold,
    s2::S2Manifold,
};
use nalgebra::{U1, U2, U3};

// Position = Direction (S2) + Range (R1)
// Tangent Dim = 2 + 1 = 3
type SphericalPos = CompositeManifold<f64, S2Manifold<f64>, EuclideanManifold<f64, U1>, U2, U1>;

// Full State = Position + Velocity (R3)
// Tangent Dim = 3 + 3 = 6
type RadarState = CompositeManifold<f64, SphericalPos, EuclideanManifold<f64, U3>, U3, U3>;
```

### Step 2: The Process Model (Physics)
We want a **Constant Velocity** model. But our state is Spherical!
Instead of converting to Cartesian, let's do "Pure Manifold" physics.

1. **Decompose Velocity**: Split 3D velocity into **Radial** (along direction) and **Tangential** (perpendicular).
2. **Update Range**: $r_{new} = r + v_{radial} \cdot dt$
3. **Update Direction**: Use `retract` to move the direction by the angular rate caused by $v_{tangential}$.

```rust
impl ManifoldProcess<RadarState, U3, f64> for ConstantVelocityModel {
    fn predict(&self, state: &RadarState, dt: f64, _: Option<&OVector<f64, U3>>) -> RadarState {
        let dir = state.first.first.as_unit_vector();
        let range = state.first.second.as_vector()[0];
        let vel = state.second.as_vector(); // 3D Cartesian velocity

        // 1. Radial Velocity
        let v_radial = vel.dot(dir.as_ref());

        // 2. Tangential Velocity
        let v_tan = vel - dir.as_ref() * v_radial;

        // 3. Update Range
        let new_range = range + v_radial * dt;

        // 4. Update Direction (Retract!)
        // Convert tangential velocity to angular rate (radians/sec)
        // We project v_tan onto the 2D local basis of the sphere
        let (e1, e2) = local_basis(dir);
        let omega = Vector2::new(v_tan.dot(&e1), v_tan.dot(&e2)) / range;
        
        // Move along the sphere surface
        let new_dir = state.first.first.retract(&(omega * dt));

        // Construct new state...
        // ...
    }
}
```

### Step 3: The UKF
The Unscented Kalman Filter (UKF) in Estima handles the rest.
- It generates sigma points in the **Tangent Space** (where it's flat).
- It **Retracts** them to the Manifold to pass them through your non-linear Process/Measurement models.
- It computes the **Fréchet Mean** (geometric average) of the results.

```rust
let ukf = UnscentedKalmanFilter::new(
    initial_state,
    initial_cov, // 6x6 Matrix (defined in Tangent Space!)
    process_model,
    q, // Process Noise
    measurement_model,
    r, // Measurement Noise
    sigma_points,
    weights,
    strategy, // How to average (FrechetMean for S2, Euclidean for others)
);
```

## 5. Why This Matters

By using `S2Manifold` for direction:
1. **No Singularities**: You can track objects passing directly overhead (the "North Pole") without math blowing up.
2. **Correct Uncertainty**: The covariance is defined on the tangent plane. A circular uncertainty looks like a circle on the sphere, not a distorted ellipse in lat/lon.
3. **Clean Code**: No `if angle < -PI { angle += 2*PI }` hacks. The `retract` and `local` functions handle topology internally.

## Next Steps
- Check out `examples/radar_ukf.rs` for the full code of this tutorial.
- Check out `examples/ahrs_ukf.rs` for orientation tracking with Quaternions.
