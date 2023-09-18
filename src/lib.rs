use numpy::ndarray::{s, Array2, ArrayView2, Zip};
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
        let mut curr = field.clone().to_owned();
        let mut next = field.clone().to_owned();
        let dx = dxdy.0.powi(2);
        let dy = dxdy.1.powi(2);
        for _ in 0..iter {
            Zip::from(next.slice_mut(s![1..-1, 1..-1]))
                .and(curr.windows((3, 3)))
                .par_for_each(|n, w| {
                    let left = &w[(0, 1)];
                    let right = &w[(2, 1)];
                    let up = &w[(1, 0)];
                    let down = &w[(1, 2)];
                    let mid = &w[(1, 1)];
                    *n = mid
                        + a * dt * ((right - 2.0 * mid + left) / dx + (down - 2.0 * mid + up) / dy);
                });
            std::mem::swap(&mut curr, &mut next);
        }
        if iter % 2 == 0 {
            next.to_owned()
        } else {
            curr.to_owned()
        }
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
