//! Error types for UKF operations.

/// Errors that can occur during UKF operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UkfError {
    /// The Cholesky decomposition of a covariance matrix failed.
    CholeskyDecompositionFailed,
    /// Weighted mean on the manifold failed to converge or produced invalid data.
    MeanComputationFailed,
}
