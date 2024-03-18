use bevy::ecs::schedule::ScheduleLabel;
use bevy_sefirot::display::{present_swapchain, setup_display, DisplayTexture};
use bevy_sefirot::MirrorGraph;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;

pub mod debug;
pub mod dither;
pub mod light;

pub mod prelude {
    pub use super::{
        add_render, Render, RenderConstants, RenderFields, RenderParameters, RenderPhase,
    };
}

#[derive(
    ScheduleLabel, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect,
)]
pub struct Render;

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct RenderGraph(pub MirrorGraph);
impl FromWorld for RenderGraph {
    fn from_world(world: &mut BevyWorld) -> Self {
        Self(MirrorGraph::from_world(Render, world))
    }
}

pub fn add_render<
    F: IntoSystem<I, N, M> + 'static,
    I: 'static,
    N: AsNodes + 'static,
    M: 'static,
>(
    f: F,
) -> impl System<In = I, Out = ()> {
    MirrorGraph::add_node::<RenderGraph, F, I, N, M>(f)
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderPhase {
    Light,
    Upscale,
    Postprocess,
    Finalize,
}

// May want to add subpixel antialiasing.
#[kernel]
fn upscale_kernel(
    device: Res<Device>,
    fields: Res<RenderFields>,
    constants: Res<RenderConstants>,
) -> Kernel<fn(Vec2<i32>, Vec2<u32>)> {
    Kernel::build(&device, &fields.screen_domain, &|el, start, offset| {
        let pos = (Vec2::expr(el.x, fields.screen_domain.height() - 1 - el.y) + offset)
            / constants.upscale_factor;
        let world_el = el.at(start + pos.cast_i32());
        let color = fields.color.expr(&world_el);
        *fields.screen_color.var(&el) = color;
    })
}

#[kernel]
fn delinearize_kernel(
    device: Res<Device>,
    fields: Res<RenderFields>,
    constants: Res<RenderConstants>,
) -> Kernel<fn()> {
    Kernel::build(&device, &fields.screen_domain, &|el| {
        *fields.screen_color.var(&el) = fields.screen_color.expr(&el).powf(1.0 / constants.gamma);
    })
}
fn delinearize() -> impl AsNodes {
    delinearize_kernel.dispatch()
}

#[kernel]
fn finalize_kernel(device: Res<Device>, fields: Res<RenderFields>) -> Kernel<fn()> {
    Kernel::build(&device, &fields.screen_domain, &|el| {
        *fields.final_color.var(&el) = fields.screen_color.expr(&el).extend(1.0);
    })
}
fn finalize() -> impl AsNodes {
    finalize_kernel.dispatch()
}
fn upscale(
    constants: Res<RenderConstants>,
    parameters: Res<RenderParameters>,
    fields: Res<RenderFields>,
) -> impl AsNodes {
    let viewport_size =
        Vector2::from(fields.screen_domain.0).cast::<f32>() / constants.upscale_factor as f32;
    let view_start = parameters.view_center - viewport_size.cast::<f32>() / 2.0;
    let start_integral = view_start.map(|x| x.floor() as i32);
    let start_fractional = view_start - start_integral.cast::<f32>();
    let offset = (start_fractional * constants.upscale_factor as f32)
        .try_cast::<u32>()
        .unwrap();
    upscale_kernel.dispatch(&Vec2::from(start_integral), &Vec2::from(offset))
}

#[derive(Default, Resource, Debug, Clone, Copy)]
pub struct RenderParameters {
    pub view_center: Vector2<f32>,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderConstants {
    pub gamma: f32,
    pub upscale_factor: u32,
}
impl Default for RenderConstants {
    fn default() -> Self {
        Self {
            gamma: 2.2,
            upscale_factor: 16,
        }
    }
}

#[derive(Resource)]
pub struct RenderFields {
    // In world-space.
    pub color: VField<Vec3<f32>, Vec2<i32>>,

    pub screen_domain: StaticDomain<2>,
    pub screen_color: VField<Vec3<f32>, Vec2<u32>>,
    // After non-linear color correction.
    // TODO: If using a bevy texture, this may not be necessary.
    final_color: VField<Vec4<f32>, Vec2<u32>>,
    _fields: FieldSet,
}

fn setup_fields(
    mut commands: Commands,
    device: Res<Device>,
    world: Res<World>,
    display: Query<&DisplayTexture>,
) {
    let display = display.single();
    let mut fields = FieldSet::new();
    let screen_domain = display.domain;
    let color = fields.create_bind("render-color", world.create_texture(&device));
    let screen_color = fields.create_bind(
        "render-screen-color",
        screen_domain.map_tex2d(device.create_tex2d(
            PixelStorage::Float4,
            screen_domain.width(),
            screen_domain.height(),
            1,
        )),
    );
    let final_color = display.color;
    commands.insert_resource(RenderFields {
        color,
        screen_domain,
        screen_color,
        final_color,
        _fields: fields,
    })
}

// TODO: Inserting this as a resource is kinda hacky.
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderPlugin {
    pub parameters: RenderParameters,
    pub constants: RenderConstants,
}
impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.parameters)
            .insert_resource(self.constants)
            .init_schedule(Render)
            .configure_sets(
                Render,
                (
                    RenderPhase::Light,
                    RenderPhase::Upscale,
                    RenderPhase::Postprocess,
                    RenderPhase::Finalize,
                )
                    .chain(),
            )
            .add_systems(Startup, init_resource::<RenderGraph>)
            .add_systems(Startup, setup_fields.after(setup_display))
            .add_systems(
                InitKernel,
                (
                    init_upscale_kernel,
                    init_delinearize_kernel,
                    init_finalize_kernel,
                ),
            )
            .add_systems(
                Render,
                (
                    add_render(upscale).in_set(RenderPhase::Upscale),
                    add_render(delinearize).in_set(RenderPhase::Postprocess),
                    add_render(finalize).in_set(RenderPhase::Finalize),
                ),
            )
            .add_systems(
                PostUpdate,
                (run_schedule(Render), execute_graph::<RenderGraph>)
                    .chain()
                    .after(present_swapchain),
            );
    }
}
