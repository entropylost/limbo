use super::*;
use crate::imf::{ImfFields, IMF_CAP};
use crate::physics::{PhysicsFields, NULL_OBJECT};

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
        let color = if object == NULL_OBJECT {
            Vec3::expr(1.0, 0.2, 0.0) * imf.value.expr(&el).as_f32() / IMF_CAP as f32
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
