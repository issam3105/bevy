#define_import_path bevy_pbr::transmission::depth_peel_bindings

#ifdef MULTISAMPLED
@group(4) @binding(0) var depth_peel_opaque_depth_texture: texture_depth_multisampled_2d;
@group(4) @binding(1) var depth_peel_previous_depth_texture: texture_depth_multisampled_2d;
#else
@group(4) @binding(0) var depth_peel_opaque_depth_texture: texture_depth_2d;
@group(4) @binding(1) var depth_peel_previous_depth_texture: texture_depth_2d;
#endif

fn depth_peel_discard(frag_coord: vec4<f32>, sample_index: u32) {
#ifdef MULTISAMPLED
    let opaque_depth = textureLoad(depth_peel_opaque_depth_texture, vec2<i32>(frag_coord.xy), i32(sample_index));
    let previous_depth = textureLoad(depth_peel_previous_depth_texture, vec2<i32>(frag_coord.xy), i32(sample_index));
#else
    let opaque_depth = textureLoad(depth_peel_opaque_depth_texture, vec2<i32>(frag_coord.xy), 0);
    let previous_depth = textureLoad(depth_peel_previous_depth_texture, vec2<i32>(frag_coord.xy), 0);
#endif

    if frag_coord.z <= opaque_depth || frag_coord.z >= previous_depth {
        discard;
    }
}
