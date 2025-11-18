use criterion::{black_box, criterion_group, criterion_main, Criterion};
use estima::manifold::{
    averaging::{ChordalMean, EuclideanMean, FrechetMean, ManifoldWeightedMean},
    euclidean::EuclideanManifold,
    quaternion::UnitQuaternionManifold,
    InitialGuess,
};
use nalgebra::{UnitQuaternion, Vector3, U3};
use rand::prelude::*;

fn benchmark_euclidean_mean(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let n_points = 100;
    let points: Vec<EuclideanManifold<f64, U3>> = (0..n_points)
        .map(|_| {
            EuclideanManifold::new(Vector3::new(
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
                rng.random_range(-10.0..10.0),
            ))
        })
        .collect();
    let weights: Vec<f64> = (0..n_points).map(|_| 1.0 / n_points as f64).collect();

    let mut group = c.benchmark_group("Euclidean Mean");
    group.bench_function("EuclideanMean", |b| {
        b.iter(|| {
            EuclideanMean
                .compute_mean(
                    black_box(&points),
                    black_box(&weights),
                    InitialGuess::First,
                )
                .unwrap()
        })
    });
    group.bench_function("FrechetMean", |b| {
        b.iter(|| {
            FrechetMean::default()
                .compute_mean(
                    black_box(&points),
                    black_box(&weights),
                    InitialGuess::First,
                )
                .unwrap()
        })
    });
    group.finish();
}

fn benchmark_quaternion_mean(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let n_points = 100;
    let points: Vec<UnitQuaternionManifold<f64>> = (0..n_points)
        .map(|_| {
            UnitQuaternionManifold::new(UnitQuaternion::from_euler_angles(
                rng.random_range(-0.5..0.5),
                rng.random_range(-0.5..0.5),
                rng.random_range(-0.5..0.5),
            ))
        })
        .collect();
    let weights: Vec<f64> = (0..n_points).map(|_| 1.0 / n_points as f64).collect();

    let mut group = c.benchmark_group("Quaternion Mean");
    group.bench_function("ChordalMean", |b| {
        b.iter(|| {
            ChordalMean
                .compute_mean(
                    black_box(&points),
                    black_box(&weights),
                    InitialGuess::First,
                )
                .unwrap()
        })
    });
    group.bench_function("FrechetMean", |b| {
        b.iter(|| {
            FrechetMean::default()
                .compute_mean(
                    black_box(&points),
                    black_box(&weights),
                    InitialGuess::First,
                )
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, benchmark_euclidean_mean, benchmark_quaternion_mean);
criterion_main!(benches);
