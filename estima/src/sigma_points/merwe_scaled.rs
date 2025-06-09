use super::traits::{SigmaPoints, SigmaPointsInPlace, UTSigmaCount};
use nalgebra::{
    base::{
        allocator::Allocator,
        dimension::{DimAdd, DimMul, U1, U2},
    },
    DefaultAllocator, DimName, OMatrix, OVector, RealField,
};

/// Unscented transform parameters as described in \[1\] generic over scalar S.
///
/// 1. E.A. Wan; R. Van Der Merwe "The unscented Kalman filter for nonlinear estimation"
#[derive(Clone, Copy, Debug)]
pub struct MerweScaled<S: RealField> {
    /// Determines the spread of the sigma points around the mean,
    /// usually set to a small positive value (e.g., 1e-3)
    pub alpha: S,
    /// Incorporating prior knowledge of the distribution
    /// (for Gaussian distributions, 2 is optimal)
    pub beta: S,
    /// Secondary scaling parameter (usually set to 0)
    pub kappa: S,
}

/// Merwe sigma points generator for dimension L and scalar S
impl<L, S> SigmaPoints<L, S> for MerweScaled<S>
where
    L: DimName + DimMul<U2>,
    <L as DimMul<U2>>::Output: DimAdd<U1>,
    UTSigmaCount<L>: DimName,
    DefaultAllocator:
        Allocator<L> + Allocator<L, L> + Allocator<UTSigmaCount<L>> + Allocator<L, UTSigmaCount<L>>,
    S: RealField + Copy,
{
    /// Number of sigma points, 2 * L + 1
    type SigmaCount = UTSigmaCount<L>;

    fn generate(
        &self,
        mean: &OVector<S, L>,
        sqrt_cov: &OMatrix<S, L, L>,
    ) -> (
        OMatrix<S, L, Self::SigmaCount>, // sigma points
        OVector<S, Self::SigmaCount>,    // mean weights
        OVector<S, Self::SigmaCount>,    // covariance weights
    ) {
        let n = S::from_usize(L::dim()).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let scale = (n + lambda).sqrt();

        let mut sigma_pts = OMatrix::<S, L, _>::zeros();
        let mut w_mean = OVector::<S, _>::zeros();
        let mut w_covar = OVector::<S, _>::zeros();

        let wm0 = lambda / (n + lambda);
        let wc0 = wm0 + (S::one() - self.alpha * self.alpha + self.beta);
        w_mean[0] = wm0;
        w_covar[0] = wc0;

        let scaled_sqrt = sqrt_cov * scale;

        sigma_pts.set_column(0, mean);

        let inv_two_n = S::one() / (S::from_usize(2).unwrap() * (n + lambda));
        for i in 0..L::dim() {
            w_mean[i + 1] = inv_two_n;
            w_covar[i + 1] = inv_two_n;
            w_mean[i + 1 + L::dim()] = inv_two_n;
            w_covar[i + 1 + L::dim()] = inv_two_n;

            let col = scaled_sqrt.column(i);

            sigma_pts.column_mut(i + 1).copy_from(&(mean + col));
            sigma_pts
                .column_mut(i + 1 + L::dim())
                .copy_from(&(mean - col));
        }

        (sigma_pts, w_mean, w_covar)
    }
}

/// Merwe sigma points inplace generator for dimension L and scalar S
impl<L, S> SigmaPointsInPlace<L, S> for MerweScaled<S>
where
    L: DimName + DimMul<U2>,
    <L as DimMul<U2>>::Output: DimAdd<U1>,
    UTSigmaCount<L>: DimName,
    S: RealField + Copy,
    DefaultAllocator:
        Allocator<L> + Allocator<L, L> + Allocator<L, UTSigmaCount<L>> + Allocator<UTSigmaCount<L>>,
{
    /// Number of sigma points, 2 * L + 1
    type SigmaCount = UTSigmaCount<L>;

    fn generate_into(
        &self,
        mean: &OVector<S, L>,
        sqrt_cov: &OMatrix<S, L, L>,
        sigma_pts: &mut OMatrix<S, L, Self::SigmaCount>,
        w_mean: &mut OVector<S, Self::SigmaCount>,
        w_covar: &mut OVector<S, Self::SigmaCount>,
    ) {
        let n = S::from_usize(L::dim()).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let scale = (n + lambda).sqrt();

        let wm0 = lambda / (n + lambda);
        let wc0 = wm0 + (S::one() - self.alpha * self.alpha + self.beta);
        w_mean[0] = wm0;
        w_covar[0] = wc0;

        let scaled_sqrt = sqrt_cov * scale;

        sigma_pts.set_column(0, mean);

        let inv = S::one() / (S::from_usize(2).unwrap() * (n + lambda));
        for i in 0..L::dim() {
            w_mean[i + 1] = inv;
            w_covar[i + 1] = inv;
            w_mean[i + 1 + L::dim()] = inv;
            w_covar[i + 1 + L::dim()] = inv;

            let col = scaled_sqrt.column(i);
            sigma_pts.column_mut(i + 1).copy_from(&(mean + col));
            sigma_pts
                .column_mut(i + 1 + L::dim())
                .copy_from(&(mean - col));
        }
    }
}
