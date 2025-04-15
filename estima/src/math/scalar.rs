///! Custom scalar implementation to abstract math over f32 and f64

pub trait Scalar:
    Copy
    + PartialOrd
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;

    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn abs(self) -> Self;
    fn epsilon() -> Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f32::recip(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline(always)]
    fn epsilon() -> Self {
        f32::EPSILON
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline(always)]
    fn recip(self) -> Self {
        f64::recip(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline(always)]
    fn epsilon() -> Self {
        f64::EPSILON
    }
}

/// Trait with helper function to convert usize used in const generics to Scalar
pub trait ToScalar {
    /// Semantic/const shorthand for 2.0
    const TWO: Self;

    fn from_usize(n: usize) -> Self;
}

impl ToScalar for f32 {
    const TWO: Self = 2.0;

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f32 // In UKF `as` is enough as practically N < 1000
    }
}

impl ToScalar for f64 {
    const TWO: Self = 2.0;

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scalar_properties<T: Scalar>(from_f64: fn(f64) -> T) {
        // Test ZERO and ONE
        assert_eq!(T::ZERO + T::ONE, from_f64(1.0));
        assert_eq!(T::ZERO, from_f64(0.0));
        assert_eq!(T::ONE, from_f64(1.0));

        // Addition, Subtraction, Multiplication, Division
        let a = from_f64(3.0);
        let b = from_f64(2.0);
        assert_eq!(a + b, from_f64(5.0));
        assert_eq!(a - b, from_f64(1.0));
        assert_eq!(a * b, from_f64(6.0));
        assert_eq!(a / b, from_f64(1.5));

        // Negation
        assert_eq!(-from_f64(5.0), from_f64(-5.0));

        // Sqrt
        let val = from_f64(9.0);
        assert!(
            (val.sqrt() - from_f64(3.0)).abs() < T::epsilon(),
            "sqrt: got={:?} expected={:?}",
            val.sqrt(),
            from_f64(3.0)
        );

        // Recip
        let rec = from_f64(2.0).recip();
        assert!(
            (rec - from_f64(0.5)).abs() < T::epsilon(),
            "recip: got={:?} expected={:?}",
            rec,
            from_f64(0.5)
        );

        // Abs
        let n = from_f64(-8.0);
        assert_eq!(n.abs(), from_f64(8.0));
        let p = from_f64(7.5);
        assert_eq!(p.abs(), p);

        // PartialOrd
        let x = from_f64(2.0);
        let y = from_f64(3.0);
        assert!(x < y, "PartialOrd: x < y");
        assert!(y > x, "PartialOrd: y > x");
        assert!(x <= x, "PartialOrd: x <= x");
        assert!(y >= x, "PartialOrd: y >= x");

        // Epsilon is small but positive
        assert!(T::epsilon() > T::ZERO);
    }

    #[test]
    fn scalar_trait_for_f32() {
        test_scalar_properties::<f32>(|v| v as f32);
    }

    #[test]
    fn scalar_trait_for_f64() {
        test_scalar_properties::<f64>(|v| v as f64);
    }

    #[test]
    fn f32_debug_printable() {
        let x = <f32 as Scalar>::ONE;
        let s = format!("{:?}", x);
        assert!(s.contains("1"));
    }

    #[test]
    fn f64_debug_printable() {
        let x = <f64 as Scalar>::ONE;
        let s = format!("{:?}", x);
        assert!(s.contains("1"));
    }

    #[test]
    fn check_scalar_trait_bounds() {
        // Compile test: f32 and f64 satisfy trait bounds
        fn assert_scalar<T: Scalar>() {}
        assert_scalar::<f32>();
        assert_scalar::<f64>();
    }
}
