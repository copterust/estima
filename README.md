# Estima

`estima` is a Rust crate for state estimation providing Square Root Unscented Kalman Filter
with manifolds.

Constant velocity model example could be run with:

```bash
cd estima && cargo run --example cv_ukf
```

## Rerun Visualization

If you have [rerun](https://rerun.io/) installed you could run:

```bash
cargo run --example cv_ukf --features rerun
```
