#[cfg(test)]
mod merwe_scaled {
    use crate::{
        sigma_points::merwe_scaled::MerweScaled,
        sigma_points::traits::{SigmaPoints, SigmaPointsInPlace, UTSigmaCount},
    };
    use approx::abs_diff_eq;
    use nalgebra::{DimName, OMatrix, OVector, U1, U2, U3, U4};

    const EPS: f64 = 1e-12;

    /// UT parameters: (scale, inv, wm0, wc0)
    fn ut_params(n: f64, alpha: f64, beta: f64, kappa: f64) -> (f64, f64, f64, f64) {
        let lambda = alpha * alpha * (n + kappa) - n;
        let scale = (n + lambda).sqrt();
        let inv = 1.0 / (2.0 * (n + lambda));
        let wm0 = lambda / (n + lambda);
        let wc0 = wm0 + (1.0 - alpha * alpha + beta);
        (scale, inv, wm0, wc0)
    }

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
                let (scale, inv, wm0, wc0) = ut_params(n, alpha, beta, kappa);

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
