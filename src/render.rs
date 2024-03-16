use bevy_sefirot::display::{present_swapchain, setup_display, DisplayTexture};
use bevy_sefirot::MirrorGraph;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct RenderGraph(pub MirrorGraph);
impl FromWorld for RenderGraph {
    fn from_world(world: &mut World) -> Self {
        RenderGraph(MirrorGraph::null(world.resource::<Device>()))
    }
}

// May also want to have smooth moving.
#[kernel]
fn upscale_colors(device: Res<Device>, fields: Res<RenderFields>) -> Kernel<fn()> {
    Kernel::build(&device, fields.screen_domain, &|mut el| {
        let mut world_el = fields.domain.index(*el / fields.upscale_factor, &el);
        let color = world_el.expr(&fields.color);
        *el.var(&fields.screen_color) = color;
    })
}

#[kernel]
fn linearize_colors(device: Res<Device>, fields: Res<RenderFields>) -> Kernel<fn()> {
    Kernel::build(&device, fields.screen_domain, &|mut el| {
        let color = el.expr(&fields.screen_color);
        let color = color.powf(1.0 / fields.gamma);
        *el.var(&fields.final_color) = color.extend(1.0);
    })
}

fn render(mut graph: ResMut<RenderGraph>) {
    graph.add((upscale_colors.dispatch(), linearize_colors.dispatch()).chain());
    graph.execute_clear();
}

fn setup_fields(
    mut commands: Commands,
    plugin: Res<RenderPlugin>,
    device: Res<Device>,
    display: Query<&DisplayTexture>,
) {
    let display = display.single();
    let mut fields = FieldSet::new();
    let screen_domain = display.domain;
    let upscale_factor = plugin.upscale_factor;
    assert!(screen_domain.width() % upscale_factor == 0);
    assert!(screen_domain.height() % upscale_factor == 0);
    let domain = StaticDomain::<2>::new(
        screen_domain.width() / upscale_factor,
        screen_domain.height() / upscale_factor,
    );
    let color = fields.create_bind(
        "render-color",
        domain.map_tex2d(device.create_tex2d(
            PixelStorage::Float4,
            domain.width(),
            domain.height(),
            1,
        )),
    );
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
        fields,
        domain,
        screen_domain,
        upscale_factor,
        gamma: plugin.gamma,
        color,
        screen_color,
        final_color,
    })
}

// TODO: Inserting this as a resource is kinda hacky.
#[derive(Debug, Resource, Clone)]
pub struct RenderPlugin {
    pub upscale_factor: u32,
    pub gamma: f32,
}
impl Default for RenderPlugin {
    fn default() -> Self {
        Self {
            upscale_factor: 16,
            gamma: 2.2,
        }
    }
}
impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.clone())
            .init_resource::<RenderGraph>()
            .add_systems(Startup, setup_fields.after(setup_display))
            .add_systems(InitKernel, (init_upscale_colors, init_linearize_colors))
            .add_systems(PostUpdate, render.before(present_swapchain));
    }
}
#[derive(Resource)]
pub struct RenderFields {
    pub fields: FieldSet,
    pub domain: StaticDomain<2>,
    pub screen_domain: StaticDomain<2>,
    pub upscale_factor: u32,
    pub gamma: f32,
    // In world-space, so it'll be upscaled.
    pub color: VField<Vec3<f32>, Vec2<u32>>,
    pub screen_color: VField<Vec3<f32>, Vec2<u32>>,
    // After non-linear color correction.
    // TODO: If using a bevy texture, this may not be necessary.
    final_color: VField<Vec4<f32>, Vec2<u32>>,
}
