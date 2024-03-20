use std::cell::Cell;

use bevy::ecs::schedule::{ExecutorKind, ScheduleLabel};
use bevy_sefirot::display::{setup_display, DisplayTexture};
use bevy_sefirot::luisa::init_kernel_system;
use bevy_sefirot::MirrorGraph;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;

pub mod debug;
pub mod dither;
pub mod light;

pub mod prelude {
    pub use super::{
        add_render, BuildPostprocess, PostprocessData, Render, RenderConstants, RenderFields,
        RenderPhase,
    };
}

#[derive(
    ScheduleLabel, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect,
)]
pub struct Render;

#[derive(
    ScheduleLabel, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect,
)]
pub struct BuildPostprocess;

pub struct PostprocessData {
    pub world_el: Element<Expr<Vec2<i32>>>,
    pub screen_pos: Expr<Vec2<u32>>,
    pub color: Var<Vec3<f32>>,
}

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
    Postprocess,
}

#[kernel(init = build_upscale_postprocess_kernel)]
fn upscale_postprocess_kernel(world: &mut BevyWorld) -> Kernel<fn(Vec2<i32>, Vec2<u32>)> {
    let device = (*world.resource::<Device>()).clone();
    let fields = world.resource::<RenderFields>();
    let screen_domain = fields.screen_domain;
    let color_field = fields.color;
    let final_color = fields.final_color;
    let constants = world.resource::<RenderConstants>();
    let scaling = constants.scaling;

    let world_cell = Cell::new(Some(world));

    Kernel::build(&device, &screen_domain, &|el, start, offset| {
        // Upscale
        // May want to add subpixel antialiasing.
        let pos = (Vec2::expr(el.x, screen_domain.height() - 1 - el.y) + offset) / scaling;
        let world_el = el.at(start + pos.cast_i32());
        let color = color_field.expr(&world_el).var();

        let data = PostprocessData {
            world_el,
            screen_pos: *el,
            color,
        };

        let world = world_cell.take().unwrap();

        world.insert_non_send_resource(data);

        world.run_schedule(BuildPostprocess);

        let data = world.remove_non_send_resource::<PostprocessData>().unwrap();

        *final_color.var(&el) = data.color.extend(1.0);
    })
}

#[tracked]
fn delinearize_pass(pixel: NonSend<PostprocessData>, constants: Res<RenderConstants>) {
    *pixel.color = pixel.color.powf(1.0 / constants.gamma);
}

fn upscale_postprocess(
    constants: Res<RenderConstants>,
    parameters: Res<RenderParameters>,
    fields: Res<RenderFields>,
) -> impl AsNodes {
    let viewport_size =
        Vector2::from(fields.screen_domain.0).cast::<f32>() / constants.scaling as f32;
    let view_start = parameters.view_center - viewport_size.cast::<f32>() / 2.0;
    let start_integral = view_start.map(|x| x.floor() as i32);
    let start_fractional = view_start - start_integral.cast::<f32>();
    let offset = (start_fractional * constants.scaling as f32)
        .try_cast::<u32>()
        .unwrap();
    upscale_postprocess_kernel.dispatch(&Vec2::from(start_integral), &Vec2::from(offset))
}

#[derive(Default, Resource, Debug, Clone, Copy)]
pub struct RenderParameters {
    pub view_center: Vector2<f32>,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderConstants {
    pub gamma: f32,
    pub scaling: u32,
}
impl Default for RenderConstants {
    fn default() -> Self {
        Self {
            gamma: 2.2,
            scaling: 12,
        }
    }
}

#[derive(Resource)]
pub struct RenderFields {
    // In world-space.
    pub color: VField<Vec3<f32>, Vec2<i32>>,
    pub screen_domain: StaticDomain<2>,
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
    let final_color = display.color;
    commands.insert_resource(RenderFields {
        color,
        screen_domain,
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
        let mut postprocess_schedule = Schedule::new(BuildPostprocess);
        postprocess_schedule.set_executor_kind(ExecutorKind::SingleThreaded);
        app.insert_resource(self.parameters)
            .insert_resource(self.constants)
            .init_schedule(Render)
            .add_schedule(postprocess_schedule)
            .configure_sets(
                Render,
                (RenderPhase::Light, RenderPhase::Postprocess).chain(),
            )
            .add_systems(Startup, init_resource::<RenderGraph>)
            .add_systems(Startup, setup_fields.after(setup_display))
            .add_systems(
                PostStartup,
                build_upscale_postprocess_kernel.after(init_kernel_system),
            )
            .add_systems(
                Update,
                run_schedule::<Render>.before(run_schedule::<WorldUpdate>),
            )
            .add_systems(HostUpdate, execute_graph::<RenderGraph>)
            .add_systems(BuildPostprocess, delinearize_pass)
            .add_systems(
                Render,
                add_render(upscale_postprocess).in_set(RenderPhase::Postprocess),
            );
    }
}
