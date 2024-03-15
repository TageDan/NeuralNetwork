use pyo3::prelude::*;

mod activation;
mod layer;
mod network;
mod perceptron;

pub use activation::Activation;
pub use layer::Layer;
pub use network::Network;
pub use perceptron::Perceptron;

#[pymodule]
fn nn_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function()
}

