use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
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
        field: ArrayView2<'_, f64>,
        dxdy: (f64, f64),
        a: f64,
        dt: f64,
        iter: u64,
    ) -> Array2<f64> {
        // TODO: Implement heat equations here
        Array2::zeros(field.raw_dim())
    }

    // Wrapper for the above pure Rust function
    #[pyfn(m)]
    #[pyo3(name = "evolve")]
    fn evolve_py<'py>(
        py: Python<'py>,
        field: PyReadonlyArray2<'py, f64>,
        dxdy: (f64, f64),
        a: f64,
        dt: f64,
        iter: u64,
    ) -> &'py PyArray2<f64> {
        let f = field.as_array();
        let res = evolve(f, dxdy, a, dt, iter);
        res.into_pyarray(py)
    }
    Ok(())
}
