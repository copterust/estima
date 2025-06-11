/// UT parameters: (scale, inv, wm0, wc0)
pub fn ut_params(n: f64, alpha: f64, beta: f64, kappa: f64) -> (f64, f64, f64, f64) {
    let lambda = alpha * alpha * (n + kappa) - n;
    let scale = (n + lambda).sqrt();
    let inv = 1.0 / (2.0 * (n + lambda));
    let wm0 = lambda / (n + lambda);
    let wc0 = wm0 + (1.0 - alpha * alpha + beta);
    (scale, inv, wm0, wc0)
}

#[cfg(test)]
mod merwe_scaled {
    use crate::{
        sigma_points::merwe_scaled::MerweScaled,
        sigma_points::traits::{SigmaPoints, SigmaPointsInPlace, UTSigmaCount},
    };
    use approx::abs_diff_eq;
    use nalgebra::{DimName, OMatrix, OVector, U1, U2, U3, U4};

    const EPS: f64 = 1e-12;

    macro_rules! test_dim {
        ($name:ident, $L:ty) => {
            #[test]
            fn $name() {
                // pick non‐trivial UT params
                let alpha = 0.6;
                let beta = 1.2;
                let kappa = 0.4;
                let ut = MerweScaled { alpha, beta, kappa };

                // build a non‐identity sqrt_cov
                let sqrt_cov = {
                    let d = <$L as DimName>::dim();
                    let mut m = OMatrix::<f64, $L, $L>::identity();
                    for i in 0..d {
                        m[(i, i)] = 1.0 + 0.1 * (i as f64 + 1.0);
                    }
                    m
                };

                // non‐trivial mean: [–1, 1, –1, …]
                let mean = OVector::<f64, $L>::from_fn(|i, _| if i % 2 == 0 { -1.0 } else { 1.0 });

                let (pts_g, wm_g, wc_g) = ut.generate(&mean, &sqrt_cov);

                let mut pts_i = OMatrix::<f64, $L, _>::zeros();
                let mut wm_i = OVector::<f64, _>::zeros();
                let mut wc_i = OVector::<f64, _>::zeros();
                ut.generate_into(&mean, &sqrt_cov, &mut pts_i, &mut wm_i, &mut wc_i);

                let cols = pts_g.ncols();
                assert_eq!(cols, UTSigmaCount::<$L>::dim());
                assert_eq!(cols, pts_i.ncols());

                for r in 0..<$L as DimName>::dim() {
                    for c in 0..cols {
                        assert!(
                            abs_diff_eq!(pts_g[(r, c)], pts_i[(r, c)], epsilon = EPS),
                            "pts mismatch at ({},{})",
                            r,
                            c
                        );
                    }
                }
                for i in 0..cols {
                    assert!(
                        abs_diff_eq!(wm_g[i], wm_i[i], epsilon = EPS),
                        "wm mismatch at {}",
                        i
                    );
                    assert!(
                        abs_diff_eq!(wc_g[i], wc_i[i], epsilon = EPS),
                        "wc mismatch at {}",
                        i
                    );
                }

                // sum-of-mean-weights == 1
                let sum_wm: f64 = wm_i.iter().copied().sum();
                assert!(abs_diff_eq!(sum_wm, 1.0, epsilon = EPS));

                let expected_wc_sum = 1.0 + (1.0 - alpha * alpha + beta);
                let sum_wc: f64 = wc_i.iter().copied().sum();
                assert!(
                    abs_diff_eq!(sum_wc, expected_wc_sum, epsilon = EPS),
                    "sum wc {} vs expected {}",
                    sum_wc,
                    expected_wc_sum
                );

                let n = <$L as DimName>::dim() as f64;
                let (scale, inv, wm0, wc0) = super::ut_params(n, alpha, beta, kappa);

                for r in 0..<$L as DimName>::dim() {
                    assert!(
                        abs_diff_eq!(pts_i[(r, 0)], mean[r], epsilon = EPS),
                        "head point mismatch at row {}",
                        r
                    );
                }
                assert!(abs_diff_eq!(wm_i[0], wm0, epsilon = EPS));
                assert!(abs_diff_eq!(wc_i[0], wc0, epsilon = EPS));

                let axis = (<$L as DimName>::dim() - 1) as usize;
                let idx = cols - 1;
                for r in 0..<$L as DimName>::dim() {
                    let expected = if r == axis {
                        mean[r] - sqrt_cov[(r, r)] * scale
                    } else {
                        mean[r]
                    };
                    assert!(
                        abs_diff_eq!(pts_i[(r, idx)], expected, epsilon = EPS),
                        "last point mismatch at ({},{})",
                        r,
                        idx
                    );
                }
                assert!(abs_diff_eq!(wm_i[idx], inv, epsilon = EPS));
                assert!(abs_diff_eq!(wc_i[idx], inv, epsilon = EPS));
            }
        };
    }

    test_dim!(dim1_general, U1);
    test_dim!(dim2_general, U2);
    test_dim!(dim3_general, U3);
    test_dim!(dim4_general, U4);
}

mod traits {
    use crate::sigma_points::UTSigmaCount;
    use nalgebra::constraint::{DimEq, ShapeConstraint};
    use nalgebra::{DimName, U1, U2, U3, U4, U5, U7, U9};

    trait AssertDimEq<A: DimName, B: DimName>
    where
        ShapeConstraint: DimEq<A, B>,
    {
        fn check();
    }

    impl<A: DimName, B: DimName> AssertDimEq<A, B> for ()
    where
        ShapeConstraint: DimEq<A, B>,
    {
        fn check() {}
    }

    #[test]
    fn ut_sigma_count() {
        <() as AssertDimEq<UTSigmaCount<U1>, U3>>::check(); // 2 × 1 + 1 = 3
        <() as AssertDimEq<UTSigmaCount<U2>, U5>>::check(); // 2 × 2 + 1 = 5
        <() as AssertDimEq<UTSigmaCount<U3>, U7>>::check(); // 2 × 3 + 1 = 7
        <() as AssertDimEq<UTSigmaCount<U4>, U9>>::check(); // 2 × 4 + 1 = 9
    }
}

#[cfg(test)]
mod unscented_transform_tests {
    use crate::{
        sigma_points::merwe_scaled::MerweScaled, sigma_points::traits::SigmaPoints,
        sigma_points::unscented_transform,
    };
    use approx::abs_diff_eq;
    use nalgebra::{OMatrix, OVector, U2, U3};

    #[test]
    fn recombine_nonidentity_covariance_u3() {
        type N = U3;
        let alpha = 0.8;
        let beta = 2.0;
        let kappa = 0.5;
        let ut = MerweScaled { alpha, beta, kappa };

        let sqrt_cov = OMatrix::<f64, N, N>::new(1.2, 0.3, 0.0, 0.0, 1.1, 0.4, 0.0, 0.0, 0.9);
        let mean = OVector::<f64, N>::new(0.5, -1.0, 2.0);

        let (pts, w_m, w_c) = ut.generate(&mean, &sqrt_cov);
        let (mean_rec, sqrt_cov_rec) = unscented_transform::<N, _, f64>(&pts, &w_m, &w_c);

        for i in 0..3 {
            assert!(abs_diff_eq!(mean_rec[i], mean[i], epsilon = 1e-12));
        }

        let mut cov_direct = OMatrix::<f64, N, N>::zeros();
        for i in 0..pts.ncols() {
            let d = pts.column(i) - mean_rec;
            cov_direct += w_c[i] * &d * d.transpose();
        }
        let cov_qr = &sqrt_cov_rec * sqrt_cov_rec.transpose();
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    abs_diff_eq!(cov_qr[(r, c)], cov_direct[(r, c)], epsilon = 1e-12),
                    "covariance mismatch at ({},{})",
                    r,
                    c
                );
            }
        }
    }

    #[test]
    fn recombine_identity_covariance_u2() {
        type N = U2;
        let alpha = 1.0;
        let beta = 2.0;
        let kappa = 0.0;
        let ut = MerweScaled { alpha, beta, kappa };

        let mean = OVector::<f64, N>::new(1.0, -2.0);
        let sqrt_cov = OMatrix::<f64, N, N>::identity();

        let (pts, w_m, w_c) = ut.generate(&mean, &sqrt_cov);

        let (mean_rec, sqrt_cov_rec) = unscented_transform::<N, _, f64>(&pts, &w_m, &w_c);

        assert!(abs_diff_eq!(mean_rec[0], mean[0], epsilon = 1e-12));
        assert!(abs_diff_eq!(mean_rec[1], mean[1], epsilon = 1e-12));

        let cov_rec = &sqrt_cov_rec * sqrt_cov_rec.transpose();
        let identity = OMatrix::<f64, N, N>::identity();
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    abs_diff_eq!(cov_rec[(r, c)], identity[(r, c)], epsilon = 1e-12),
                    "cov_rec[{r},{c}] = {}, expected {}",
                    cov_rec[(r, c)],
                    identity[(r, c)]
                );
            }
        }
    }
    #[test]
    fn recombine_manual_u1_two_points() {
        use crate::sigma_points::unscented_transform;
        use approx::abs_diff_eq;
        use nalgebra::{OMatrix, OVector, U1, U2};

        type N = U1;
        type Σ = U2;

        let pts = OMatrix::<f64, N, Σ>::new(0.0, 2.0);

        let w_m = OVector::<f64, Σ>::new(0.5, 0.5);
        let w_c = OVector::<f64, Σ>::new(0.5, 0.5);

        let (mean_rec, sqrt_cov_rec) = unscented_transform::<N, Σ, f64>(&pts, &w_m, &w_c);

        assert!(abs_diff_eq!(mean_rec[0], 1.0, epsilon = 1e-12));

        let cov_rec = sqrt_cov_rec[(0, 0)] * sqrt_cov_rec[(0, 0)];
        assert!(abs_diff_eq!(cov_rec, 1.0, epsilon = 1e-12));
    }
}

#[cfg(test)]
mod merwe_scaled_extra {
    use crate::sigma_points::merwe_scaled::MerweScaled;
    use crate::sigma_points::traits::SigmaPoints;
    use crate::sigma_points::unscented_transform;
    use approx::abs_diff_eq;
    use nalgebra::{OMatrix, OVector, U1, U2, U3};

    const EPS: f64 = 1e-8;

    #[test]
    fn extreme_ut_parameters_degenerate_u1() {
        type N = U1;
        let alpha = 1e-6;
        let beta = 2.0;
        let kappa = -1.0;
        let ut = MerweScaled { alpha, beta, kappa };

        let sqrt_cov = OMatrix::<f64, N, N>::identity();
        let mean = OVector::<f64, N>::zeros();
        let (pts, wm, wc) = ut.generate(&mean, &sqrt_cov);

        // all points collapse to mean
        for i in 0..pts.ncols() {
            assert!(abs_diff_eq!(pts[(0, i)], 0.0, epsilon = EPS));
        }
        let sum_wm: f64 = wm.iter().copied().sum();
        assert!(abs_diff_eq!(sum_wm, 1.0, epsilon = EPS));
        let sum_wc: f64 = wc.iter().copied().sum();
        assert!(abs_diff_eq!(
            sum_wc,
            1.0 - alpha * alpha + beta,
            epsilon = EPS
        ));
    }

    #[test]
    fn extreme_ut_parameters_degenerate_u2() {
        type N = U2;
        let alpha = 1e-6;
        let beta = 2.0;
        let kappa = -2.0;
        let ut = MerweScaled { alpha, beta, kappa };

        let sqrt_cov = OMatrix::<f64, N, N>::identity();
        let mean = OVector::<f64, N>::zeros();
        let (pts, wm, wc) = ut.generate(&mean, &sqrt_cov);

        for r in 0..2 {
            for i in 0..pts.ncols() {
                assert!(abs_diff_eq!(pts[(r, i)], 0.0, epsilon = EPS));
            }
        }
        let sum_wm: f64 = wm.iter().copied().sum();
        assert!(abs_diff_eq!(sum_wm, 1.0, epsilon = EPS));
        let sum_wc: f64 = wc.iter().copied().sum();
        assert!(abs_diff_eq!(
            sum_wc,
            1.0 - alpha * alpha + beta,
            epsilon = EPS
        ));
    }

    #[test]
    fn extreme_ut_parameters_degenerate_u3() {
        type N = U3;
        let alpha = 1e-6;
        let beta = 2.0;
        let kappa = -3.0;
        let ut = MerweScaled { alpha, beta, kappa };

        let sqrt_cov = OMatrix::<f64, N, N>::identity();
        let mean = OVector::<f64, N>::zeros();
        let (pts, wm, wc) = ut.generate(&mean, &sqrt_cov);

        for r in 0..3 {
            for i in 0..pts.ncols() {
                assert!(abs_diff_eq!(pts[(r, i)], 0.0, epsilon = EPS));
            }
        }
        let sum_wm: f64 = wm.iter().copied().sum();
        assert!(abs_diff_eq!(sum_wm, 1.0, epsilon = EPS));
        let sum_wc: f64 = wc.iter().copied().sum();
        assert!(abs_diff_eq!(
            sum_wc,
            1.0 - alpha * alpha + beta,
            epsilon = EPS
        ));
    }

    #[test]
    fn ill_conditioned_covariance_reconstruction() {
        type N = U2;
        let alpha = 0.9;
        let beta = 2.0;
        let kappa = 0.5;
        let ut = MerweScaled { alpha, beta, kappa };

        let eps = 1e-12;
        let diag = OMatrix::<f64, N, N>::new(1.0, 0.0, 0.0, 1.0 + eps);
        let mean = OVector::<f64, N>::new(0.0, 0.0);

        let (pts, w_m, w_c) = ut.generate(&mean, &diag);
        let (_mean_rec, sqrt_cov_rec) = unscented_transform::<N, _, f64>(&pts, &w_m, &w_c);

        let cov_rec = &sqrt_cov_rec * sqrt_cov_rec.transpose();
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    abs_diff_eq!(cov_rec[(r, c)], diag[(r, c)], epsilon = 1e-10),
                    "Mismatch at ({},{}): {} vs {}",
                    r,
                    c,
                    cov_rec[(r, c)],
                    diag[(r, c)]
                );
            }
        }
    }

    #[test]
    fn verify_scaling_offsets_u3() {
        type N = U3;
        let alpha = 0.7;
        let beta = 2.0;
        let kappa = 0.3;
        let ut = MerweScaled { alpha, beta, kappa };

        let sqrt_cov = OMatrix::<f64, N, N>::identity();
        let mean = OVector::<f64, N>::new(1.0, -2.0, 0.5);
        let (pts, _, wc) = ut.generate(&mean, &sqrt_cov);

        let n = 3.0;
        let (scale, _, _, wc0) = super::ut_params(n, alpha, beta, kappa);

        assert!(abs_diff_eq!(wc[0], wc0, epsilon = EPS));
        let idx_pos = 1 + 2;
        let offset_vec = pts.column(idx_pos) - mean;
        for i in 0..3 {
            let expected = if i == 2 { scale } else { 0.0 };
            assert!(
                abs_diff_eq!(offset_vec[i], expected, epsilon = EPS),
                "Offset mismatch at {}: {} vs {}",
                i,
                offset_vec[i],
                expected
            );
        }
    }

    #[test]
    fn linear_transform_recombine() {
        use nalgebra::Matrix2;
        type N = U2;
        let alpha = 0.8;
        let beta = 1.0;
        let kappa = 0.0;
        let ut = MerweScaled { alpha, beta, kappa };

        let mean = OVector::<f64, N>::new(2.0, -1.0);
        let sqrt_cov = OMatrix::<f64, N, N>::identity() * 0.5;

        let A = Matrix2::new(1.0, 2.0, -0.5, 0.3);
        let b = OVector::<f64, N>::new(0.2, -0.1);

        let (pts, w_m, w_c) = ut.generate(&mean, &sqrt_cov);

        let mut pts2 = pts.clone();
        for i in 0..pts2.ncols() {
            let y = &A * pts2.column(i) + &b;
            pts2.set_column(i, &y);
        }

        let (mean2, sqrt_cov2) = unscented_transform::<N, _, f64>(&pts2, &w_m, &w_c);
        let mean_expected = &A * mean + &b;
        let cov_expected = A * (&sqrt_cov * &sqrt_cov.transpose()) * A.transpose();

        for i in 0..2 {
            assert!(abs_diff_eq!(mean2[i], mean_expected[i], epsilon = EPS));
        }
        let cov2 = &sqrt_cov2 * sqrt_cov2.transpose();
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    abs_diff_eq!(cov2[(r, c)], cov_expected[(r, c)], epsilon = 1e-10),
                    "Lin-cov mismatch at ({},{}): {} vs {}",
                    r,
                    c,
                    cov2[(r, c)],
                    cov_expected[(r, c)]
                );
            }
        }
    }
}
