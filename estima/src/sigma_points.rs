use nalgebra::base::dimension::{DimAdd, DimMul, U1, U2};

/// The compile‐time number of sigma points (2 × N + 1) for the standard UT.
pub type UTSigmaCount<N> = <<N as DimMul<U2>>::Output as DimAdd<U1>>::Output;

#[cfg(test)]
mod test {
    use super::UTSigmaCount;
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
    fn test_ut_sigma_count() {
        <() as AssertDimEq<UTSigmaCount<U1>, U3>>::check(); // 2 × 1 + 1 = 3
        <() as AssertDimEq<UTSigmaCount<U2>, U5>>::check(); // 2 × 2 + 1 = 5
        <() as AssertDimEq<UTSigmaCount<U3>, U7>>::check(); // 2 × 3 + 1 = 7
        <() as AssertDimEq<UTSigmaCount<U4>, U9>>::check(); // 2 × 4 + 1 = 9
    }
}
