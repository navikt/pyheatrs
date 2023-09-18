use numpy::ndarray::{ArrayD, ArrayViewD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

/// A Python module for the heat equations implemented in Rust
///
/// To complete this module properly using [`ndarray`](https://docs.rs/ndarray/0.15.6/ndarray/) will be required.
#[pymodule]
fn pyheatrs(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Evolve the heat equation over the given field
    ///
    /// # Arguments
    /// - `field` - Immutable array view of the field to evolve
    /// - `dxdy` - Delta for X and Y dimensions
    /// - `a` - Diffusion constant to use for the evolution
    /// - `dt` - Time delta for the evolutions
    /// - `iter` - Number of iterations to perform
    fn evolve(
        field: ArrayViewD<'_, f64>,
        dxdy: (f64, f64),
        a: f64,
        dt: f64,
        iter: u64,
    ) -> ArrayD<f64> {
        // TODO: Implement heat equations here
        ArrayD::zeros(field.raw_dim())
    }

    // Wrapper for the above pure Rust function
    #[pyfn(m)]
    #[pyo3(name = "evolve")]
    fn evolve_py<'py>(
        py: Python<'py>,
        field: PyReadonlyArrayDyn<'py, f64>,
        dxdy: (f64, f64),
        a: f64,
        dt: f64,
        iter: u64,
    ) -> &'py PyArrayDyn<f64> {
        let f = field.as_array();
        let res = evolve(f, dxdy, a, dt, iter);
        res.into_pyarray(py)
    }
    Ok(())
}
