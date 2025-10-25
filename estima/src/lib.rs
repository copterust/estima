//! Estimation algoritms in Rust

pub mod filter;
pub mod kf;
pub mod manifold;
pub mod sigma_points;
pub mod sr_ukf;

/// Trait defining a state
pub trait State<const DIM: usize> {
    type Scalar: Default + Copy;

    /// Convert state to a flat array
    fn as_array(&self) -> &[Self::Scalar; DIM];

    /// Update state from a flat array
    fn from_array(data: &[Self::Scalar; DIM]) -> Self;
}

/// Trait defining a process model
pub trait ProcessModel<S: State<DIM>, const DIM: usize> {
    /// Predict the next state based on the current state, time step, and optional control inputs
    fn predict(&self, state: &S, dt: S::Scalar, control: Option<&[S::Scalar]>) -> S;
}

/// Trait defining a measurement model for UKF
pub trait MeasurementModel<S: State<DIM>, Measurement, const DIM: usize> {
    /// Generate a measurement from the current state
    fn measure(&self, state: &S) -> Measurement;

    /// Compute the residual between a predicted measurement and an actual measurement
    fn residual(&self, predicted: &Measurement, measurement: &Measurement) -> Measurement;
}

/// Builder for UKF instances
pub struct UKFBuilder<S: State<DIM>, Measurement, Process, MeasurementModel, const DIM: usize> {
    state: Option<S>,
    process: Option<Process>,
    measurement_model: Option<MeasurementModel>,
    _marker: core::marker::PhantomData<Measurement>,
}

impl<S, Measurement, Process, MM, const DIM: usize> Default
    for UKFBuilder<S, Measurement, Process, MM, DIM>
where
    S: State<DIM>,
    Process: ProcessModel<S, DIM>,
    MM: MeasurementModel<S, Measurement, DIM>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S, Measurement, Process, MM, const DIM: usize> UKFBuilder<S, Measurement, Process, MM, DIM>
where
    S: State<DIM>,
    Process: ProcessModel<S, DIM>,
    MM: MeasurementModel<S, Measurement, DIM>,
{
    /// Create a new UKF builder
    pub fn new() -> Self {
        Self {
            state: None,
            process: None,
            measurement_model: None,
            _marker: core::marker::PhantomData,
        }
    }

    /// Set the initial state
    pub fn with_state(mut self, state: S) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the process model
    pub fn with_process(mut self, process: Process) -> Self {
        self.process = Some(process);
        self
    }

    /// Set the measurement model
    pub fn with_measurement(mut self, measurement_model: MM) -> Self {
        self.measurement_model = Some(measurement_model);
        self
    }

    /// Build the UKF instance
    pub fn build(self) -> UKF<S, Measurement, Process, MM, DIM> {
        UKF {
            state: self.state.expect("State must be set"),
            process: self.process.expect("Process model must be set"),
            measurement_model: self
                .measurement_model
                .expect("Measurement model must be set"),
            _marker: core::marker::PhantomData,
        }
    }
}

/// UKF structure
pub struct UKF<State, Measurement, Process, MeasModel, const DIM: usize> {
    state: State,
    process: Process,
    measurement_model: MeasModel,
    _marker: core::marker::PhantomData<Measurement>,
}

impl<S, Measurement, Process, MeasModel, const DIM: usize>
    UKF<S, Measurement, Process, MeasModel, DIM>
where
    S: State<DIM>,
    Process: ProcessModel<S, DIM>,
    MeasModel: MeasurementModel<S, Measurement, DIM>,
{
    /// Perform a prediction step without control inputs
    pub fn predict(&mut self, dt: S::Scalar) {
        self.state = self.process.predict(&self.state, dt, None);
    }

    /// Perform a prediction step with control inputs
    pub fn predict_with_control(&mut self, dt: S::Scalar, control: Option<&[S::Scalar]>) {
        self.state = self.process.predict(&self.state, dt, control);
    }

    /// Perform an update step with a measurement
    pub fn update(&mut self, measurement: &Measurement) {
        let predicted_measurement = self.measurement_model.measure(&self.state);
        let residual = self
            .measurement_model
            .residual(&predicted_measurement, measurement);
        // TODO
        self.apply_residual(residual);
    }

    /// Get the current state
    pub fn state(&self) -> &S {
        &self.state
    }

    fn apply_residual(&mut self, _residual: Measurement) {
        // TODO
    }
}
