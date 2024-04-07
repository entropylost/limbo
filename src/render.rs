use std::cell::Cell as StdCell;

use bevy::ecs::schedule::{ExecutorKind, ScheduleLabel};
use bevy_sefirot::display::{setup_display, DisplayTexture};
use bevy_sefirot::luisa::init_kernel_system;
use bevy_sefirot::MirrorGraph;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;
use crate::world::UpdateGraph;

pub mod agx;
pub mod debug;
pub mod dither;
pub mod light;

pub mod prelude {
    pub use super::{
        add_render, BuildPostprocess, PostprocessData, PostprocessPhase, Render, RenderConstants,
        RenderFields, RenderPhase,
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
    Postprocess,
}

#[derive(Default, Resource, Debug, Clone, Copy)]
pub struct RenderParameters {
    pub view_center: Vector2<f32>,
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct RenderConstants {
    pub scaling: u32,
}
impl Default for RenderConstants {
    fn default() -> Self {
        Self { scaling: 12 }
    }
}

#[derive(Resource)]
pub struct RenderFields {
    // In world-space.
    pub color: VField<Vec3<f32>, Cell>,
    pub screen_domain: StaticDomain<2>,
    final_color: VEField<Vec4<f32>, Vec2<u32>>,
    _fields: FieldSet,
}

fn setup_render(
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

#[derive(
    ScheduleLabel, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect,
)]
pub struct BuildPostprocess;

pub struct PostprocessData {
    pub cell: Element<Expr<Vec2<i32>>>,
    pub subcell_pos: Expr<Vec2<u32>>,
    pub screen_pos: Expr<Vec2<u32>>,
    pub color: Var<Vec3<f32>>,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PostprocessPhase {
    Tonemap,
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

    let world_cell = StdCell::new(Some(world));

    Kernel::build(&device, &screen_domain, &|pixel, start, offset| {
        // Upscale
        // May want to add subpixel antialiasing.
        let pos = Vec2::expr(pixel.x, screen_domain.height() - 1 - pixel.y) + offset;
        let subcell_pos = pos % scaling;
        let pos = pos / scaling;
        let cell = pixel.at(start + pos.cast_i32());
        let color = color_field.expr(&cell).var();

        let data = PostprocessData {
            cell,
            subcell_pos,
            screen_pos: *pixel,
            color,
        };

        let world = world_cell.take().unwrap();

        world.insert_non_send_resource(data);

        world.run_schedule(BuildPostprocess);

        let data = world.remove_non_send_resource::<PostprocessData>().unwrap();

        *final_color.var(&pixel) = data.color.extend(1.0);
    })
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
            .add_systems(Startup, setup_render.after(setup_display))
            .add_systems(
                PostStartup,
                build_upscale_postprocess_kernel.after(init_kernel_system),
            )
            .add_systems(
                Update,
                run_schedule::<Render>
                    .after(run_schedule::<WorldUpdate>)
                    .before(execute_graph::<UpdateGraph>),
            )
            .add_systems(
                Update,
                execute_graph::<RenderGraph>.after(execute_graph::<UpdateGraph>),
            )
            .add_systems(
                Render,
                add_render(upscale_postprocess).in_set(RenderPhase::Postprocess),
            );
    }
}
