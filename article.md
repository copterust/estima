Filtering and estimation are bread and butter for engineers. A popular way to
estimate the true state of a system from noisy sensor data is the Kalman filter,
but many real-world systems are nonlinear: for example, conversions between polar
and Cartesian coordinates, and rigid-body rotations, are ubiquitous in robotics
and avionics.

For robotics we want reliability and precision; central to that is accurate state
estimation. The classic Kalman filter works great when things behave in a mostly
straight-line way, but the real world does not: the Extended Kalman Filter
handles that by doing a bunch of derivative math to pretend things are straight
locally, while the Unscented Kalman Filter skips that tendous math and instead
pushes a few what-ifs through the real model to know how to update its guess.

Another complementary approach is to respect the geometry of the state. You do
it with something called "manifold". Manifold-aware filters use maps to move
between the manifold and its tangent space, apply Kalman-style updates in the
tangent (which is Euclidean), and then map back -- preserving constraints and
avoiding singularities.

You already know how to live on manifold. Just look around, everything is so
flat! But if you zoom out a bit there are hills, valeys and who knows, maybe
Earth itself is spherical!

Geometry was born measuring flat Earth. Lines were straight and connect two
points via shortest distance. Eucledean space is physical and intuitive. Things
are different when you draw lines on a sphere. Good that we could zoom close
enough on a point on manifold: things would get back Eucledean like they seems
around you. So manifolds are geometeic spaces (or smooth, continous shapes) that
looks "flat" (Eucledean) in any of it's points. Look close enough and circle is
a line. This local "flatness" allow application of Kalman-style estimation
correctly without pretending rotations live in familiar R^3 space.

Question: can we implement UKF and manifold-based filtering in Rust, safely and
efficiently? Absolutely. Rust's strong types and generics make it a great fit
for encoding manifold constraints, producing safe, no-allocation (no_std)
implementations, and writing numerically robust variants for production-grade
performance.

## References
* https://arxiv.org/abs/2102.03804
