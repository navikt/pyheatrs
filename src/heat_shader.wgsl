// Input array which defines the current heat field
@group(0) @binding(0) var<storage, read> current: array<f32>;
// Output array which defines the evolved heat field
@group(0) @binding(1) var<storage, read_write> next: array<f32>;

struct Params {
    dxdy2: vec2<f32>,
    diffusion: f32,
    dt: f32,
    row_length: u32,
}
// Distance between points in the X and Y direction
@group(0) @binding(2) var<storage, read> params: Params;

fn idx(x: u32, y: u32) -> u32 {
    return y * params.row_length + x;
}

fn evolve(x: u32, y: u32) -> f32 {
    let left: f32 = current[idx(x - 1u, y)];
    let right: f32 = current[idx(x + 1u, y)];
    let up: f32 = current[idx(x, y - 1u)];
    let down: f32 = current[idx(x, y + 1u)];
    let mid: f32 = current[idx(x, y)];

    let dx: f32 = (right - 2.0 * mid + left) / params.dxdy2.x;
    let dy: f32 = (down - 2.0 * mid + up) / params.dxdy2.y;

    return mid + params.diffusion * params.dt * (dx + dy);
}

@compute
@workgroup_size(1,1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // We add 1 to both indices so that we index valid values only and it simplifies kernel
    // dispatch
    let x: u32 = global_id.x + 1u;
    let y: u32 = global_id.y + 1u;
    next[idx(x, y)] = evolve(x, y);
}


