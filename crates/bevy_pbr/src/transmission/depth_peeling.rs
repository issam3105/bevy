use crate::{
    get_mesh_instance_world_from_local, init_material_pipeline, DrawMesh, MaterialFragmentShader,
    MaterialPipeline, MaterialVertexShader, MeshInputUniform, MeshPipeline, MeshPipelineKey,
    MeshPipelineSystems, MeshUniform, PreparedMaterial, RenderMaterialInstances,
    RenderMeshInstances, ScreenSpaceTransmission, SetMaterialBindGroup, SetMeshBindGroup,
    SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup, ViewKeyCache, ViewTransmissionTexture,
    MATERIAL_BIND_GROUP_INDEX,
};

use alloc::sync::Arc;
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::{Camera, Camera3d};
use bevy_core_pipeline::core_3d::{TransparentSortingInfo3d, CORE_3D_DEPTH_FORMAT};
use bevy_ecs::{
    entity::{Entity, EntityHash},
    prelude::*,
    query::With,
    schedule::IntoScheduleConfigs,
    system::{
        lifetimeless::{Read, SRes},
        SystemParam,
    },
};
use bevy_image::ToExtents;
use bevy_material::{
    key::{ErasedMaterialPipelineKey, ErasedMeshPipelineKey},
    MaterialProperties,
};
use bevy_mesh::{Mesh, Mesh3d, MeshVertexBufferLayoutRef};
use bevy_render::{
    batching::gpu_preprocessing::BatchedInstanceBuffers,
    camera::ExtractedCamera,
    erased_render_asset::ErasedRenderAssets,
    mesh::RenderMesh,
    render_asset::RenderAssets,
    render_phase::{
        sort_phase_system, AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId,
        DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand, RenderCommandResult,
        SetItemPipeline, SortedPhaseItem, SortedRenderPhasePlugin, TrackedRenderPass,
        ViewSortedRenderPhases,
    },
    render_resource::{
        binding_types::texture_2d, BindGroup, BindGroupEntries, BindGroupLayout,
        BindGroupLayoutDescriptor, BindGroupLayoutEntries, CachedRenderPipelineId,
        CompareFunction, DepthBiasState, DepthStencilState, Extent3d, FragmentState, LoadOp,
        MultisampleState, Operations, PipelineCache, PrimitiveState,
        RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        ShaderStages, SpecializedMeshPipeline, SpecializedMeshPipelineError,
        SpecializedMeshPipelines, StoreOp, TextureDescriptor, TextureDimension, TextureSampleType,
        TextureUsages, VertexState,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    sync_world::MainEntity,
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity, ViewDepthTexture, ViewTarget},
    Extract, ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderStartup, RenderSystems,
};
use bevy_shader::{load_shader_library, Shader, ShaderDefVal};
use core::{
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};
use indexmap::IndexMap;
use tracing::error;

pub(super) struct TransmissionDepthPeelingPlugin;

impl Plugin for TransmissionDepthPeelingPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "depth_prepass.wgsl");
        load_shader_library!(app, "depth_peel_bindings.wgsl");

        if app.get_sub_app(RenderApp).is_none() {
            return;
        }

        app.add_plugins((
            SortedRenderPhasePlugin::<DepthPeelDepth3d, MeshPipeline>::new(
                RenderDebugFlags::default(),
            ),
            SortedRenderPhasePlugin::<DepthPeelColor3d, MeshPipeline>::new(
                RenderDebugFlags::default(),
            ),
        ));

        let render_app = app
            .get_sub_app_mut(RenderApp)
            .expect("render app should exist");

        render_app
            .init_resource::<DrawFunctions<DepthPeelDepth3d>>()
            .init_resource::<DrawFunctions<DepthPeelColor3d>>()
            .init_resource::<ViewSortedRenderPhases<DepthPeelDepth3d>>()
            .init_resource::<ViewSortedRenderPhases<DepthPeelColor3d>>()
            .init_resource::<SpecializedMeshPipelines<DepthPeelDepthPipeline>>()
            .init_resource::<SpecializedMeshPipelines<DepthPeelColorPipeline>>()
            .init_resource::<DepthPeelCurrentLayer>()
            .add_render_command::<DepthPeelDepth3d, DrawDepthPeelDepth>()
            .add_render_command::<DepthPeelColor3d, DrawDepthPeelColor>()
            .add_systems(
                RenderStartup,
                init_depth_peel_pipeline
                    .after(MeshPipelineSystems)
                    .after(init_material_pipeline),
            )
            .add_systems(ExtractSchedule, extract_depth_peel_camera_phases)
            .add_systems(
                Render,
                (
                    prepare_depth_peel_textures.in_set(RenderSystems::PrepareResources),
                    prepare_depth_peel_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    queue_depth_peeled_meshes.in_set(RenderSystems::QueueMeshes),
                    sort_phase_system::<DepthPeelDepth3d>.in_set(RenderSystems::PhaseSort),
                    sort_phase_system::<DepthPeelColor3d>.in_set(RenderSystems::PhaseSort),
                ),
            );
    }
}

pub(super) fn configure_depth_peeling_view_targets(
    mut cameras: Query<(&ScreenSpaceTransmission, &mut Camera3d)>,
) {
    for (transmission, mut camera_3d) in &mut cameras {
        if transmission.depth_peeling {
            camera_3d.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
        }
    }
}

#[derive(Resource, Default)]
pub(super) struct DepthPeelCurrentLayer(AtomicUsize);

#[derive(Component)]
pub(super) struct ViewDepthPeelTextures {
    initial: CachedTexture,
    layers: Vec<CachedTexture>,
}

#[derive(Component)]
struct ViewDepthPeelBindGroups {
    bind_groups: Vec<BindGroup>,
}

#[derive(Resource)]
struct DepthPeelPipeline {
    mesh_pipeline: MeshPipeline,
    material_pipeline: MaterialPipeline,
    depth_shader: Handle<Shader>,
    depth_view_layout: BindGroupLayoutDescriptor,
    depth_view_bind_group_layout: BindGroupLayout,
}

fn init_depth_peel_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    material_pipeline: Res<MaterialPipeline>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::FRAGMENT,
        (
            texture_2d(TextureSampleType::Depth),
            texture_2d(TextureSampleType::Depth),
        ),
    );
    let depth_view_bind_group_layout =
        render_device.create_bind_group_layout("transmission_depth_peel_view_layout", &entries);
    let depth_view_layout =
        BindGroupLayoutDescriptor::new("transmission_depth_peel_view_layout", &entries);

    commands.insert_resource(DepthPeelPipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        material_pipeline: material_pipeline.clone(),
        depth_shader: load_embedded_asset!(asset_server.as_ref(), "depth_prepass.wgsl"),
        depth_view_layout,
        depth_view_bind_group_layout,
    });
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct DepthPeelDepthPipelineKey {
    mesh_key: MeshPipelineKey,
}

impl SpecializedMeshPipeline for DepthPeelDepthPipeline {
    type Key = DepthPeelDepthPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut shader_defs = vec![
            "MESH_PIPELINE".into(),
            "VERTEX_OUTPUT_INSTANCE_INDEX".into(),
            "TRANSMISSION_DEPTH_PEEL_PASS".into(),
        ];
        if key.mesh_key.msaa_samples() > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }

        let vertex_buffer_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        let view_layout = self.mesh_pipeline.get_view_layout(key.mesh_key.into());
        let mesh_layout = self.mesh_pipeline.mesh_layouts.model_only.clone();

        Ok(RenderPipelineDescriptor {
            label: Some("transmission_depth_peel_depth_pipeline".into()),
            layout: vec![
                view_layout.main_layout,
                view_layout.binding_array_layout,
                mesh_layout,
                view_layout.empty_layout,
                self.depth_view_layout.clone(),
            ],
            vertex: VertexState {
                shader: self.depth_shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_buffer_layout],
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: self.depth_shader.clone(),
                shader_defs,
                targets: vec![],
                ..Default::default()
            }),
            primitive: PrimitiveState {
                topology: key.mesh_key.primitive_topology(),
                strip_index_format: key.mesh_key.strip_index_format(),
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.mesh_key.msaa_samples(),
                ..Default::default()
            },
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        })
    }
}

struct DepthPeelDepthPipeline {
    mesh_pipeline: MeshPipeline,
    depth_shader: Handle<Shader>,
    depth_view_layout: BindGroupLayoutDescriptor,
}

struct DepthPeelColorPipeline {
    material_pipeline: MaterialPipeline,
    properties: Arc<MaterialProperties>,
}

impl SpecializedMeshPipeline for DepthPeelColorPipeline {
    type Key = ErasedMaterialPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let concrete_mesh_key: MeshPipelineKey = key.mesh_key.downcast();
        let mut descriptor = self
            .material_pipeline
            .mesh_pipeline
            .specialize(concrete_mesh_key, layout)?;

        descriptor.vertex.shader_defs.push(ShaderDefVal::UInt(
            "MATERIAL_BIND_GROUP".into(),
            MATERIAL_BIND_GROUP_INDEX as u32,
        ));
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.push(ShaderDefVal::UInt(
                "MATERIAL_BIND_GROUP".into(),
                MATERIAL_BIND_GROUP_INDEX as u32,
            ));
        }
        if let Some(vertex_shader) = self.properties.get_shader(MaterialVertexShader) {
            descriptor.vertex.shader = vertex_shader;
        }
        if let Some(fragment_shader) = self.properties.get_shader(MaterialFragmentShader) {
            descriptor.fragment.as_mut().unwrap().shader = fragment_shader;
        }
        descriptor
            .layout
            .insert(3, self.properties.material_layout.as_ref().unwrap().clone());
        if let Some(specialize) = self.properties.user_specialize {
            specialize(
                &self.material_pipeline as &dyn core::any::Any,
                &mut descriptor,
                layout,
                key,
            )?;
        }
        if self.properties.bindless {
            descriptor.vertex.shader_defs.push("BINDLESS".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("BINDLESS".into());
            }
        }
        if let Some(depth_stencil) = descriptor.depth_stencil.as_mut() {
            depth_stencil.depth_write_enabled = Some(false);
            depth_stencil.depth_compare = Some(CompareFunction::Equal);
        }

        Ok(descriptor)
    }
}

impl From<&DepthPeelPipeline> for DepthPeelDepthPipeline {
    fn from(pipeline: &DepthPeelPipeline) -> Self {
        Self {
            mesh_pipeline: pipeline.mesh_pipeline.clone(),
            depth_shader: pipeline.depth_shader.clone(),
            depth_view_layout: pipeline.depth_view_layout.clone(),
        }
    }
}

pub(super) struct DepthPeelDepth3d {
    distance: f32,
    entity: (Entity, MainEntity),
    pipeline: CachedRenderPipelineId,
    draw_function: DrawFunctionId,
    batch_range: Range<u32>,
    extra_index: PhaseItemExtraIndex,
    indexed: bool,
}

pub(super) struct DepthPeelColor3d {
    sorting_info: TransparentSortingInfo3d,
    distance: f32,
    entity: (Entity, MainEntity),
    pipeline: CachedRenderPipelineId,
    draw_function: DrawFunctionId,
    batch_range: Range<u32>,
    extra_index: PhaseItemExtraIndex,
    indexed: bool,
}

macro_rules! impl_depth_peel_phase_item {
    ($ty:ty) => {
        impl PhaseItem for $ty {
            const AUTOMATIC_BATCHING: bool = false;

            fn entity(&self) -> Entity {
                self.entity.0
            }

            fn main_entity(&self) -> MainEntity {
                self.entity.1
            }

            fn draw_function(&self) -> DrawFunctionId {
                self.draw_function
            }

            fn batch_range(&self) -> &Range<u32> {
                &self.batch_range
            }

            fn batch_range_mut(&mut self) -> &mut Range<u32> {
                &mut self.batch_range
            }

            fn extra_index(&self) -> PhaseItemExtraIndex {
                self.extra_index.clone()
            }

            fn batch_range_and_extra_index_mut(
                &mut self,
            ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
                (&mut self.batch_range, &mut self.extra_index)
            }
        }

        impl CachedRenderPipelinePhaseItem for $ty {
            fn cached_pipeline(&self) -> CachedRenderPipelineId {
                self.pipeline
            }
        }
    };
}

impl_depth_peel_phase_item!(DepthPeelDepth3d);
impl_depth_peel_phase_item!(DepthPeelColor3d);

impl SortedPhaseItem for DepthPeelDepth3d {
    type SortKey = bevy_math::FloatOrd;

    fn sort_key(&self) -> Self::SortKey {
        bevy_math::FloatOrd(self.distance)
    }

    fn sort(items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>) {
        items.sort_by_key(|_, item| item.sort_key());
    }

    fn recalculate_sort_keys(
        _items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        _view: &ExtractedView,
    ) {
    }

    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl SortedPhaseItem for DepthPeelColor3d {
    type SortKey = bevy_math::FloatOrd;

    fn sort_key(&self) -> Self::SortKey {
        bevy_math::FloatOrd(self.distance)
    }

    fn sort(items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>) {
        items.sort_by_key(|_, item| item.sort_key());
    }

    fn recalculate_sort_keys(
        items: &mut IndexMap<(Entity, MainEntity), Self, EntityHash>,
        view: &ExtractedView,
    ) {
        let rangefinder = view.rangefinder3d();
        for item in items.values_mut() {
            item.distance = item.sorting_info.sort_distance(&rangefinder);
        }
    }

    fn indexed(&self) -> bool {
        self.indexed
    }
}

type DrawDepthPeelDepth = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    SetDepthPeelViewBindGroup<4>,
    DrawMesh,
);

type DrawDepthPeelColor = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<MATERIAL_BIND_GROUP_INDEX>,
    DrawMesh,
);

struct SetDepthPeelViewBindGroup<const I: usize>;

impl<const I: usize> RenderCommand<DepthPeelDepth3d> for SetDepthPeelViewBindGroup<I> {
    type Param = SRes<DepthPeelCurrentLayer>;
    type ViewQuery = Read<ViewDepthPeelBindGroups>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &DepthPeelDepth3d,
        bind_groups: &'w ViewDepthPeelBindGroups,
        _entity: Option<()>,
        current_layer: bevy_ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let layer = current_layer.into_inner().0.load(Ordering::Relaxed);
        let Some(bind_group) = bind_groups.bind_groups.get(layer) else {
            return RenderCommandResult::Failure("missing transmission depth peel bind group");
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

fn extract_depth_peel_camera_phases(
    mut depth_phases: ResMut<ViewSortedRenderPhases<DepthPeelDepth3d>>,
    mut color_phases: ResMut<ViewSortedRenderPhases<DepthPeelColor3d>>,
    cameras: Extract<Query<(Entity, &Camera, &ScreenSpaceTransmission), With<Camera3d>>>,
    mut live_entities: Local<Vec<RetainedViewEntity>>,
) {
    live_entities.clear();
    for (main_entity, camera, settings) in &cameras {
        if !camera.is_active || !settings.depth_peeling || settings.steps == 0 {
            continue;
        }

        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);
        depth_phases.prepare_for_new_frame(retained_view_entity);
        color_phases.prepare_for_new_frame(retained_view_entity);
        live_entities.push(retained_view_entity);
    }

    depth_phases.retain(|view_entity, _| live_entities.contains(view_entity));
    color_phases.retain(|view_entity, _| live_entities.contains(view_entity));
}

fn prepare_depth_peel_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &ScreenSpaceTransmission)>,
) {
    for (entity, camera, settings) in &views {
        if !settings.depth_peeling || settings.steps == 0 {
            commands
                .entity(entity)
                .remove::<(ViewDepthPeelTextures, ViewDepthPeelBindGroups)>();
            continue;
        }

        let Some(size) = camera.physical_target_size else {
            continue;
        };

        let descriptor = TextureDescriptor {
            label: Some("transmission_depth_peel_layer_texture"),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CORE_3D_DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let initial = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("transmission_depth_peel_initial_previous_texture"),
                ..descriptor.clone()
            },
        );
        let layers = (0..settings.steps)
            .map(|_| texture_cache.get(&render_device, descriptor.clone()))
            .collect();

        commands
            .entity(entity)
            .insert(ViewDepthPeelTextures { initial, layers });
    }
}

fn prepare_depth_peel_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<DepthPeelPipeline>,
    views: Query<(Entity, &ViewDepthTexture, &ViewDepthPeelTextures)>,
) {
    for (entity, opaque_depth, textures) in &views {
        let mut bind_groups = Vec::with_capacity(textures.layers.len());
        for layer in 0..textures.layers.len() {
            let previous_view = if layer == 0 {
                &textures.initial.default_view
            } else {
                &textures.layers[layer - 1].default_view
            };
            bind_groups.push(render_device.create_bind_group(
                Some("transmission_depth_peel_view_bind_group"),
                &pipeline.depth_view_bind_group_layout,
                &BindGroupEntries::sequential((opaque_depth.view(), previous_view)),
            ));
        }

        commands
            .entity(entity)
            .insert(ViewDepthPeelBindGroups { bind_groups });
    }
}

#[derive(SystemParam)]
struct QueueDepthPeelParams<'w, 's> {
    depth_draw_functions: Res<'w, DrawFunctions<DepthPeelDepth3d>>,
    color_draw_functions: Res<'w, DrawFunctions<DepthPeelColor3d>>,
    depth_pipelines: ResMut<'w, SpecializedMeshPipelines<DepthPeelDepthPipeline>>,
    color_pipelines: ResMut<'w, SpecializedMeshPipelines<DepthPeelColorPipeline>>,
    pipeline_cache: Res<'w, PipelineCache>,
    depth_peel_pipeline: Res<'w, DepthPeelPipeline>,
    render_meshes: Res<'w, RenderAssets<RenderMesh>>,
    render_materials: Res<'w, ErasedRenderAssets<PreparedMaterial>>,
    render_mesh_instances: Res<'w, RenderMeshInstances>,
    render_material_instances: Res<'w, RenderMaterialInstances>,
    maybe_batched_instance_buffers:
        Option<Res<'w, BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>>,
    depth_phases: ResMut<'w, ViewSortedRenderPhases<DepthPeelDepth3d>>,
    color_phases: ResMut<'w, ViewSortedRenderPhases<DepthPeelColor3d>>,
    views: Query<
        'w,
        's,
        (
            &'static ExtractedView,
            &'static RenderVisibleEntities,
            &'static ScreenSpaceTransmission,
        ),
    >,
    view_key_cache: Res<'w, ViewKeyCache>,
}

fn queue_depth_peeled_meshes(params: QueueDepthPeelParams) {
    let QueueDepthPeelParams {
        depth_draw_functions,
        color_draw_functions,
        mut depth_pipelines,
        mut color_pipelines,
        pipeline_cache,
        depth_peel_pipeline,
        render_meshes,
        render_materials,
        render_mesh_instances,
        render_material_instances,
        maybe_batched_instance_buffers,
        mut depth_phases,
        mut color_phases,
        views,
        view_key_cache,
    } = params;

    let draw_depth = depth_draw_functions.read().id::<DrawDepthPeelDepth>();
    let draw_color = color_draw_functions.read().id::<DrawDepthPeelColor>();

    for (view, visible_entities, settings) in &views {
        if !settings.depth_peeling || settings.steps == 0 {
            continue;
        }

        let (Some(depth_phase), Some(color_phase)) = (
            depth_phases.get_mut(&view.retained_view_entity),
            color_phases.get_mut(&view.retained_view_entity),
        ) else {
            continue;
        };

        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };
        let Some(render_visible_mesh_entities) = visible_entities.get::<Mesh3d>() else {
            continue;
        };

        depth_phase.clear();
        color_phase.clear();

        for (_render_entity, visible_entity) in render_visible_mesh_entities.iter_visible() {
            let Some(material_instance) = render_material_instances.instances.get(visible_entity)
            else {
                continue;
            };
            let Some(material) = render_materials.get(material_instance.asset_id) else {
                continue;
            };
            if !material.properties.reads_view_transmission_texture {
                continue;
            }
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id()) else {
                continue;
            };

            let mut mesh_key = view_key
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
                | material.properties.mesh_pipeline_key_bits.downcast();
            mesh_key |= MeshPipelineKey::from_primitive_topology_and_strip_index(
                mesh.primitive_topology(),
                mesh.index_format(),
            );
            mesh_key |= MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE;

            let depth_pipeline = DepthPeelDepthPipeline::from(depth_peel_pipeline.as_ref());
            let depth_pipeline_id = match depth_pipelines.specialize(
                &pipeline_cache,
                &depth_pipeline,
                DepthPeelDepthPipelineKey { mesh_key },
                &mesh.layout,
            ) {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            let color_pipeline = DepthPeelColorPipeline {
                material_pipeline: depth_peel_pipeline.material_pipeline.clone(),
                properties: material.properties.clone(),
            };
            let color_key = ErasedMaterialPipelineKey {
                type_id: material_instance.asset_id.type_id(),
                mesh_key: ErasedMeshPipelineKey::new(mesh_key),
                material_key: material.properties.material_key.clone(),
            };
            let color_pipeline_id = match color_pipelines.specialize(
                &pipeline_cache,
                &color_pipeline,
                color_key,
                &mesh.layout,
            ) {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            let sorting_info = TransparentSortingInfo3d::Sorted {
                mesh_center: get_mesh_instance_world_from_local(
                    *visible_entity,
                    mesh_instance.current_uniform_index,
                    &render_mesh_instances,
                    maybe_batched_instance_buffers.as_deref(),
                )
                .transform_point3(mesh.aabb_center),
                depth_bias: material.properties.depth_bias,
            };

            depth_phase.add(DepthPeelDepth3d {
                distance: 0.0,
                entity: (Entity::PLACEHOLDER, *visible_entity),
                pipeline: depth_pipeline_id,
                draw_function: draw_depth,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: mesh.indexed(),
            });
            color_phase.add(DepthPeelColor3d {
                sorting_info,
                distance: 0.0,
                entity: (Entity::PLACEHOLDER, *visible_entity),
                pipeline: color_pipeline_id,
                draw_function: draw_color,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: mesh.indexed(),
            });
        }
    }
}

pub(super) fn depth_peeling_pass_3d(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ScreenSpaceTransmission,
        &ViewDepthPeelTextures,
        Option<&ViewTransmissionTexture>,
    )>,
    depth_phases: Res<ViewSortedRenderPhases<DepthPeelDepth3d>>,
    color_phases: Res<ViewSortedRenderPhases<DepthPeelColor3d>>,
    current_layer: Res<DepthPeelCurrentLayer>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();
    let (camera, extracted_view, target, settings, textures, transmission) = view.into_inner();
    if !settings.depth_peeling || settings.steps == 0 || textures.layers.is_empty() {
        return;
    }

    let Some(depth_phase) = depth_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };
    let Some(color_phase) = color_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };

    {
        let _clear = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("transmission_depth_peel_clear_initial_previous"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &textures.initial.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    for (layer, texture) in textures.layers.iter().enumerate() {
        current_layer.0.store(layer, Ordering::Relaxed);
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("transmission_depth_peel_depth_layer"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &texture.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Err(err) = depth_phase.render(&mut pass, world, view_entity) {
            error!("Error rendering transmission depth peel depth phase: {err:?}");
        }
    }

    let Some(transmission) = transmission else {
        return;
    };
    let Some(size) = camera.physical_target_size else {
        return;
    };

    for texture in textures.layers.iter().rev() {
        ctx.command_encoder().copy_texture_to_texture(
            target.main_texture().as_image_copy(),
            transmission.texture.as_image_copy(),
            size.to_extents(),
        );
        let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("transmission_depth_peel_color_layer"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &texture.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Err(err) = color_phase.render(&mut pass, world, view_entity) {
            error!("Error rendering transmission depth peel color phase: {err:?}");
        }
    }
}
