//! Error types for the Kalman filter implementations.

/// Errors that can occur during UKF operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UkfError {
    /// The Cholesky decomposition of a covariance matrix failed, indicating it was not positive-definite.
    CholeskyDecompositionFailed,
    /// The computation of a weighted mean on a manifold failed to converge.
    MeanComputationFailed,
}
