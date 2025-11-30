//! Error types for UKF operations.

/// Errors that can occur during UKF operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UkfError {
    /// The Cholesky decomposition of a covariance matrix failed.
    CholeskyDecompositionFailed,
    /// Weighted mean on the manifold failed to converge or produced invalid data.
    MeanComputationFailed,
}

impl core::fmt::Display for UkfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CholeskyDecompositionFailed => {
                write!(f, "Cholesky decomposition of covariance matrix failed")
            }
            Self::MeanComputationFailed => write!(f, "Mean computation on manifold failed"),
        }
    }
}

impl core::error::Error for UkfError {}
