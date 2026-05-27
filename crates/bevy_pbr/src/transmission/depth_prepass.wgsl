#import bevy_pbr::{
    mesh_functions,
    view_transformations::position_world_to_clip,
}

#import bevy_pbr::transmission::depth_peel_bindings::depth_peel_discard

struct Vertex {
    @location(0) position: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local = mesh_functions::get_world_from_local(in.instance_index);
    let world_position = world_from_local * vec4<f32>(in.position, 1.0);
    out.position = position_world_to_clip(world_position.xyz);
    return out;
}

@fragment
fn fragment(in: VertexOutput) {
    depth_peel_discard(in.position, 0u);
}
