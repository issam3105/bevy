@if(PBR_SHARED_MATERIAL_SAMPLERS)
import package::render::mesh_view_bindings::{
    pbr_linear_clamp_sampler,
    pbr_linear_repeat_sampler,
    pbr_nearest_clamp_sampler,
    pbr_nearest_repeat_sampler,
};

const SHARED_SAMPLER_MODE_MASK: u32 = 3u;
const BASE_COLOR_SAMPLER_MODE_SHIFT: u32 = 0u;
const EMISSIVE_SAMPLER_MODE_SHIFT: u32 = 2u;
const METALLIC_ROUGHNESS_SAMPLER_MODE_SHIFT: u32 = 4u;
const OCCLUSION_SAMPLER_MODE_SHIFT: u32 = 6u;
const NORMAL_MAP_SAMPLER_MODE_SHIFT: u32 = 8u;
const DEPTH_MAP_SAMPLER_MODE_SHIFT: u32 = 10u;
const ANISOTROPY_SAMPLER_MODE_SHIFT: u32 = 12u;
const SPECULAR_TRANSMISSION_SAMPLER_MODE_SHIFT: u32 = 14u;
const THICKNESS_SAMPLER_MODE_SHIFT: u32 = 16u;
const DIFFUSE_TRANSMISSION_SAMPLER_MODE_SHIFT: u32 = 18u;
const CLEARCOAT_SAMPLER_MODE_SHIFT: u32 = 20u;
const CLEARCOAT_ROUGHNESS_SAMPLER_MODE_SHIFT: u32 = 22u;
const CLEARCOAT_NORMAL_SAMPLER_MODE_SHIFT: u32 = 24u;
const SPECULAR_SAMPLER_MODE_SHIFT: u32 = 26u;
const SPECULAR_TINT_SAMPLER_MODE_SHIFT: u32 = 28u;

fn material_sampler_mode(sampler_modes: u32, shift: u32) -> u32 {
    return (sampler_modes >> shift) & SHARED_SAMPLER_MODE_MASK;
}

/// Samples a material texture with the canonical shared sampler selected by `mode`.
@if(PBR_SHARED_MATERIAL_SAMPLERS) {
fn shared_sample_bias(texture: texture_2d<f32>, uv: vec2<f32>, bias: f32, mode: u32) -> vec4<f32> {
    switch mode {
        case 1u: { return textureSampleBias(texture, pbr_linear_repeat_sampler, uv, bias); }
        case 2u: { return textureSampleBias(texture, pbr_nearest_clamp_sampler, uv, bias); }
        case 3u: { return textureSampleBias(texture, pbr_nearest_repeat_sampler, uv, bias); }
        default: { return textureSampleBias(texture, pbr_linear_clamp_sampler, uv, bias); }
    }
}

fn shared_sample_grad(texture: texture_2d<f32>, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, mode: u32) -> vec4<f32> {
    switch mode {
        case 1u: { return textureSampleGrad(texture, pbr_linear_repeat_sampler, uv, ddx, ddy); }
        case 2u: { return textureSampleGrad(texture, pbr_nearest_clamp_sampler, uv, ddx, ddy); }
        case 3u: { return textureSampleGrad(texture, pbr_nearest_repeat_sampler, uv, ddx, ddy); }
        default: { return textureSampleGrad(texture, pbr_linear_clamp_sampler, uv, ddx, ddy); }
    }
}

fn shared_sample_level(texture: texture_2d<f32>, uv: vec2<f32>, level: f32, mode: u32) -> vec4<f32> {
    switch mode {
        case 1u: { return textureSampleLevel(texture, pbr_linear_repeat_sampler, uv, level); }
        case 2u: { return textureSampleLevel(texture, pbr_nearest_clamp_sampler, uv, level); }
        case 3u: { return textureSampleLevel(texture, pbr_nearest_repeat_sampler, uv, level); }
        default: { return textureSampleLevel(texture, pbr_linear_clamp_sampler, uv, level); }
    }
}
}
