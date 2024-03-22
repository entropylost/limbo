use super::prelude::*;
pub use crate::prelude::*;
use crate::world::flow::FlowFields;
use crate::world::imf::ImfFields;
use crate::world::physics::{self, PhysicsFields, NULL_OBJECT};

#[kernel]
fn color_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
    render: Res<RenderFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let color = if physics.object.expr(&cell) != NULL_OBJECT
            || physics.next_object.expr(&cell) != NULL_OBJECT
        {
            Vec3::expr(1.0, 0.0, 0.0)
        } else {
            Vec3::expr(0.0, 0.0, 0.0)
        };
        *render.color.var(&cell) = color;
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
