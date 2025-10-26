use super::traits::{SigmaPoints, SigmaPointsGenerated, SigmaPointsInPlace, UTSigmaCount};
use super::weights::UTWeights;
use nalgebra::{
    base::{
        allocator::Allocator,
        dimension::{DimAdd, DimMul, U1, U2},
    },
    Cholesky, DefaultAllocator, DimName, OMatrix, OVector, RealField,
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

impl<S: RealField> MerweScaled<S> {
    /// Create a new MerweScaled sigma point generator.
    ///
    /// # Arguments
    /// * `alpha` - Spread of sigma points (typically 1e-3 to 1)
    /// * `beta` - Prior knowledge parameter (2 is optimal for Gaussian)
    /// * `kappa` - Secondary scaling (typically 0 or 3-n)
    pub fn new(alpha: S, beta: S, kappa: S) -> Self {
        Self { alpha, beta, kappa }
    }
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
        sqrt_cov: &Cholesky<S, L>,
    ) -> SigmaPointsGenerated<S, L, <Self as SigmaPoints<L, S>>::SigmaCount> {
        let dim = L::dim();
        let n = S::from_usize(dim).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let n_lambda = n + lambda;
        let scale = if n_lambda.abs() < S::default_epsilon() {
            S::default_epsilon().sqrt()
        } else {
            n_lambda.sqrt()
        };

        let weights =
            UTWeights::<UTSigmaCount<L>, S>::from_merwe(dim, self.alpha, self.beta, self.kappa);
        let scaled_sqrt = sqrt_cov.l() * scale;

        let mut sigma_pts = OMatrix::<S, L, Self::SigmaCount>::zeros();
        sigma_pts.set_column(0, mean);

        for i in 0..dim {
            let col = scaled_sqrt.column(i);
            sigma_pts.column_mut(i + 1).copy_from(&(mean + col));
            sigma_pts.column_mut(i + 1 + dim).copy_from(&(mean - col));
        }

        let UTWeights { w_mean, w_covar } = weights;
        SigmaPointsGenerated {
            sigma_points: sigma_pts,
            mean_weights: w_mean,
            covariance_weights: w_covar,
        }
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
        sqrt_cov: &Cholesky<S, L>,
        sigma_pts: &mut OMatrix<S, L, Self::SigmaCount>,
        w_mean: &mut OVector<S, Self::SigmaCount>,
        w_covar: &mut OVector<S, Self::SigmaCount>,
    ) {
        let dim = L::dim();
        let n = S::from_usize(dim).unwrap();
        let lambda = self.alpha * self.alpha * (n + self.kappa) - n;
        let n_lambda = n + lambda;
        let scale = if n_lambda.abs() < S::default_epsilon() {
            S::default_epsilon().sqrt()
        } else {
            n_lambda.sqrt()
        };

        let weights =
            UTWeights::<UTSigmaCount<L>, S>::from_merwe(dim, self.alpha, self.beta, self.kappa);
        w_mean.copy_from(&weights.w_mean);
        w_covar.copy_from(&weights.w_covar);

        let scaled_sqrt = sqrt_cov.l() * scale;

        sigma_pts.set_column(0, mean);

        for i in 0..dim {
            let col = scaled_sqrt.column(i);
            sigma_pts.column_mut(i + 1).copy_from(&(mean + col));
            sigma_pts.column_mut(i + 1 + dim).copy_from(&(mean - col));
        }
    }
}

impl<S: RealField + Copy> MerweScaled<S> {
    /// Convenience helper to compute UT weights for dimension `L`.
    pub fn weights<L>(&self) -> UTWeights<UTSigmaCount<L>, S>
    where
        L: DimName + DimMul<U2>,
        <L as DimMul<U2>>::Output: DimAdd<U1>,
        <<L as DimMul<U2>>::Output as DimAdd<U1>>::Output: DimName,
        DefaultAllocator: Allocator<UTSigmaCount<L>>,
    {
        UTWeights::from_merwe(L::dim(), self.alpha, self.beta, self.kappa)
    }
}
