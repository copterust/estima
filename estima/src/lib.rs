//! Definitions for UKF

/// Trait defining a process model
pub trait ProcessModel<State> {
    /// Predict the next state based on the current state, time step, and optional control inputs
    fn predict(&self, state: &State, dt: f32, control: Option<&[f32]>) -> State;
}

/// Trait defining a measurement model for UKF
pub trait MeasurementModel<State, Measurement> {
    /// Generate a measurement from the current state
    fn measure(&self, state: &State) -> Measurement;

    /// Compute the residual between a predicted measurement and an actual measurement
    fn residual(&self, predicted: Measurement, measurement: Measurement) -> Measurement;
}

/// Builder for UKF instances
pub struct UKFBuilder<State, Measurement, Process, MeasurementModel> {
    state: Option<State>,
    process: Option<Process>,
    measurement_model: Option<MeasurementModel>,
    _marker: core::marker::PhantomData<Measurement>,
}

impl<State, Measurement, Process, MeasModel> UKFBuilder<State, Measurement, Process, MeasModel>
where
    Process: ProcessModel<State>,
    MeasModel: MeasurementModel<State, Measurement>,
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
    pub fn with_state(mut self, state: State) -> Self {
        self.state = Some(state);
        self
    }

    /// Set the process model
    pub fn with_process(mut self, process: Process) -> Self {
        self.process = Some(process);
        self
    }

    /// Set the measurement model
    pub fn with_measurement(mut self, measurement_model: MeasModel) -> Self {
        self.measurement_model = Some(measurement_model);
        self
    }

    /// Build the UKF instance
    pub fn build(self) -> UKF<State, Measurement, Process, MeasModel> {
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
pub struct UKF<State, Measurement, Process, MeasModel> {
    state: State,
    process: Process,
    measurement_model: MeasModel,
    _marker: core::marker::PhantomData<Measurement>,
}

impl<State, Measurement, Process, MeasModel> UKF<State, Measurement, Process, MeasModel>
where
    Process: ProcessModel<State>,
    MeasModel: MeasurementModel<State, Measurement>,
{
    /// Perform a prediction step without control inputs
    pub fn predict(&mut self, dt: f32) {
        self.state = self.process.predict(&self.state, dt, None);
    }

    /// Perform a prediction step with control inputs
    pub fn predict_with_control(&mut self, dt: f32, control: &[f32]) {
        self.state = self.process.predict(&self.state, dt, Some(control));
    }

    /// Perform an update step with a measurement
    pub fn update(&mut self, measurement: Measurement) {
        let predicted_measurement = self.measurement_model.measure(&self.state);
        let residual = self
            .measurement_model
            .residual(predicted_measurement, measurement);
        // TODO
        self.apply_residual(residual);
    }

    /// Get the current state
    pub fn state(&self) -> &State {
        &self.state
    }

    fn apply_residual(&mut self, _residual: Measurement) {
        // TODO
    }
}
