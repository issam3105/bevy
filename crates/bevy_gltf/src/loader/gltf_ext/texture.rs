use bevy_image::{ImageAddressMode, ImageFilterMode, ImageSamplerDescriptor};
use bevy_math::Affine2;

use gltf::{
    json::extensions::texture::TextureTransform as JsonTextureTransform,
    texture::{Info, MagFilter, MinFilter, Texture, TextureTransform, WrappingMode},
};
use serde_json::Value;

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

pub(crate) fn json_texture_transform_to_affine2(
    texture_transform: &JsonTextureTransform,
) -> Affine2 {
    Affine2::from_scale_angle_translation(
        texture_transform.scale.0.into(),
        -texture_transform.rotation.0,
        texture_transform.offset.0.into(),
    )
}

pub(crate) fn texture_info_tex_coord(info: &Info) -> u32 {
    info.texture_transform()
        .and_then(|texture_transform| texture_transform.tex_coord())
        .unwrap_or_else(|| info.tex_coord())
}

pub(crate) fn texture_info_transform(info: &Info) -> Affine2 {
    info.texture_transform()
        .map(texture_transform_to_affine2)
        .unwrap_or(Affine2::IDENTITY)
}

#[cfg(any(
    feature = "pbr_anisotropy_texture",
    feature = "pbr_specular_textures",
    feature = "pbr_multi_layer_material_textures"
))]
pub(crate) fn json_texture_info_tex_coord(info: &gltf::json::texture::Info) -> u32 {
    info.extensions
        .as_ref()
        .and_then(|extensions| extensions.texture_transform.as_ref())
        .and_then(|texture_transform| texture_transform.tex_coord)
        .unwrap_or(info.tex_coord)
}

#[cfg(any(
    feature = "pbr_anisotropy_texture",
    feature = "pbr_specular_textures",
    feature = "pbr_multi_layer_material_textures"
))]
pub(crate) fn json_texture_info_transform(info: &gltf::json::texture::Info) -> Affine2 {
    info.extensions
        .as_ref()
        .and_then(|extensions| extensions.texture_transform.as_ref())
        .map(json_texture_transform_to_affine2)
        .unwrap_or(Affine2::IDENTITY)
}

pub(crate) fn texture_transform_from_extension_value(
    default_tex_coord: u32,
    texture_transform: Option<&Value>,
) -> (u32, Affine2) {
    texture_transform
        .and_then(|texture_transform| {
            serde_json::from_value::<JsonTextureTransform>(texture_transform.clone()).ok()
        })
        .map(|texture_transform| {
            (
                texture_transform.tex_coord.unwrap_or(default_tex_coord),
                json_texture_transform_to_affine2(&texture_transform),
            )
        })
        .unwrap_or((default_tex_coord, Affine2::IDENTITY))
}
