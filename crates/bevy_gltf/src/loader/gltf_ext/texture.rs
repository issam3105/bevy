use bevy_image::{ImageAddressMode, ImageFilterMode, ImageSamplerDescriptor};
use bevy_math::Affine2;

use gltf::texture::{MagFilter, MinFilter, Texture, TextureTransform, WrappingMode};
use serde_json::{Map, Value};

/// Extracts the texture sampler data from the glTF [`Texture`].
pub(crate) fn texture_sampler(
    texture: &Texture<'_>,
    default_sampler: &ImageSamplerDescriptor,
) -> ImageSamplerDescriptor {
    let gltf_sampler = texture.sampler();
    let mut sampler = default_sampler.clone();

    sampler.address_mode_u = address_mode(&gltf_sampler.wrap_s());
    sampler.address_mode_v = address_mode(&gltf_sampler.wrap_t());

    // Shouldn't parse filters when anisotropic filtering is on, because trilinear is then required by wgpu.
    // We also trust user to have provided a valid sampler.
    if sampler.anisotropy_clamp == 1 {
        if let Some(mag_filter) = gltf_sampler.mag_filter().map(|mf| match mf {
            MagFilter::Nearest => ImageFilterMode::Nearest,
            MagFilter::Linear => ImageFilterMode::Linear,
        }) {
            sampler.mag_filter = mag_filter;
        }
        if let Some(min_filter) = gltf_sampler.min_filter().map(|mf| match mf {
            MinFilter::Nearest
            | MinFilter::NearestMipmapNearest
            | MinFilter::NearestMipmapLinear => ImageFilterMode::Nearest,
            MinFilter::Linear | MinFilter::LinearMipmapNearest | MinFilter::LinearMipmapLinear => {
                ImageFilterMode::Linear
            }
        }) {
            sampler.min_filter = min_filter;
        }
        if let Some(mipmap_filter) = gltf_sampler.min_filter().map(|mf| match mf {
            MinFilter::Nearest
            | MinFilter::Linear
            | MinFilter::NearestMipmapNearest
            | MinFilter::LinearMipmapNearest => ImageFilterMode::Nearest,
            MinFilter::NearestMipmapLinear | MinFilter::LinearMipmapLinear => {
                ImageFilterMode::Linear
            }
        }) {
            sampler.mipmap_filter = mipmap_filter;
        }
    }
    sampler
}

pub(crate) fn address_mode(wrapping_mode: &WrappingMode) -> ImageAddressMode {
    match wrapping_mode {
        WrappingMode::ClampToEdge => ImageAddressMode::ClampToEdge,
        WrappingMode::Repeat => ImageAddressMode::Repeat,
        WrappingMode::MirroredRepeat => ImageAddressMode::MirrorRepeat,
    }
}

pub(crate) fn texture_transform_to_affine2(texture_transform: TextureTransform) -> Affine2 {
    Affine2::from_scale_angle_translation(
        texture_transform.scale().into(),
        -texture_transform.rotation(),
        texture_transform.offset().into(),
    )
}

/// Parses a `KHR_texture_transform` extension from a raw JSON extensions map.
///
/// Used for texture types (e.g. normal map, occlusion) whose high-level gltf
/// types don't expose `texture_transform()` directly.
pub(crate) fn texture_transform_from_extensions(
    extensions: Option<&Map<String, Value>>,
) -> Affine2 {
    let Some(ext_map) = extensions else {
        return Affine2::IDENTITY;
    };
    let Some(khr) = ext_map.get("KHR_texture_transform") else {
        return Affine2::IDENTITY;
    };
    let offset = khr
        .get("offset")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            Some([
                arr.first()?.as_f64()? as f32,
                arr.get(1)?.as_f64()? as f32,
            ])
        })
        .unwrap_or([0.0, 0.0]);
    let rotation = khr
        .get("rotation")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let scale = khr
        .get("scale")
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            Some([
                arr.first()?.as_f64()? as f32,
                arr.get(1)?.as_f64()? as f32,
            ])
        })
        .unwrap_or([1.0, 1.0]);
    Affine2::from_scale_angle_translation(scale.into(), -rotation, offset.into())
}
