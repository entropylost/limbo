use super::prelude::*;
pub use crate::prelude::*;
use crate::world::imf::ImfFields;
use crate::world::physics::{PhysicsFields, NULL_OBJECT};

#[kernel]
fn color_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
    render: Res<RenderFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let object = physics.object.expr(&el);
        let color = if true {
            Vec3::expr(1.0, (imf.object.expr(&el) + 1).cast_f32(), 0.0) * imf.mass.expr(&el)
        } else {
            Vec3::splat_expr(1.0)
        };
        *render.color.var(&el) = color;
    })
}

fn color() -> impl AsNodes {
    color_kernel.dispatch()
}

pub struct DebugPlugin;
impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(InitKernel, init_color_kernel)
            .add_systems(Render, add_render(color).in_set(RenderPhase::Light));
    }
}
