use bytemuck::{Pod, Zeroable};
use numpy::ndarray::{s, Array2, ArrayBase, ArrayView2, Zip};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GPUParams {
    dxdy2: [f32; 2],
    diffusion: f32,
    dt: f32,
    row_length: u32,
    _padding: u32,
}

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
            next
        } else {
            curr
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

    #[pyfn(m)]
    #[pyo3(name = "evolve_gpu")]
    fn evolve_gpu_py<'py>(
        py: Python<'py>,
        field: PyReadonlyArray2<'py, f32>,
        dxdy: (f32, f32),
        a: f32,
        dt: f32,
        iter: u64,
    ) -> PyResult<&'py PyArray2<f32>> {
        let f = field.as_array();
        let res = pollster::block_on(evolve_gpu(f, dxdy, a, dt, iter));
        res.map(|r| r.into_pyarray(py))
    }

    async fn evolve_gpu(
        field: ArrayView2<'_, f32>,
        dxdy: (f32, f32),
        a: f32,
        dt: f32,
        iter: u64,
    ) -> PyResult<Array2<f32>> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or(PyRuntimeError::new_err("Request GPU adapter failed"))?;
        // If the selected adapter is not a GPU we error out to assuage expectations
        let info = adapter.get_info();
        match info.device_type {
            wgpu::DeviceType::Other | wgpu::DeviceType::Cpu | wgpu::DeviceType::VirtualGpu => {
                println!("Adapter name: {}", info.name);
                println!("Adapter backend: {:?}", info.backend);
                println!("Adapter device: {:?}", info.device_type);
                return Err(PyRuntimeError::new_err(
                    "Adapter request did not return a GPU!",
                ));
            }
            _ => {}
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|_| PyRuntimeError::new_err("Request GPU device failed"))?;
        // Convert our input matrix into a contiguous array in C-style, this is needed when passing
        // the data to the GPU as the GPU has little notion of arbitrary size matrices
        let cont = field.as_standard_layout();
        let cont_slice = cont.as_slice().unwrap();
        let size = std::mem::size_of_val(cont_slice) as wgpu::BufferAddress;

        // Instantiate buffers on the GPU, we need 3 buffers. One will be a staging buffer which we
        // will use to copy data from the GPU back to the CPU. One will contain the data from CPU
        // and be a read-only buffer on the GPU. The last will be the GPU result which contains the
        // evolved data from the read-only buffer
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let current_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Current"),
            contents: bytemuck::cast_slice(cont_slice),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let next_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Next"),
            contents: bytemuck::cast_slice(cont_slice),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        // There are a few other variables that are needed for the shader, here we use buffers and
        // instantiate them right away
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("param buffer"),
            contents: bytemuck::bytes_of(&GPUParams {
                dxdy2: [dxdy.0.powi(2), dxdy.1.powi(2)],
                diffusion: a,
                dt,
                row_length: field.shape()[1] as u32,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sh_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Heat equation shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("heat_shader.wgsl"))),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &sh_module,
            entry_point: "main",
        });
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: current_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });
        let shape = field.shape();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for _ in 0..iter {
            {
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(shape[1] as u32 - 2, shape[0] as u32 - 2, 1);
            }
            // Copy next back into current ready for the next round
            encoder.copy_buffer_to_buffer(&next_buffer, 0, &current_buffer, 0, size);
        }
        encoder.copy_buffer_to_buffer(&next_buffer, 0, &staging_buffer, 0, size);
        queue.submit(Some(encoder.finish()));

        let result_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        result_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        device.poll(wgpu::Maintain::Wait);
        if let Some(Ok(())) = receiver.receive().await {
            let data = result_slice.get_mapped_range();
            let slice: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            let result = ArrayBase::from_shape_vec(field.raw_dim(), slice);

            // Tell the GPU to release the memory associated with the staging buffer
            drop(data);
            staging_buffer.unmap();

            result.map_err(|_| PyRuntimeError::new_err("Could not convert GPU data to 2D array"))
        } else {
            Err(PyRuntimeError::new_err("Could not run on GPU!"))
        }
    }
    Ok(())
}
