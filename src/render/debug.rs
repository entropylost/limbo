use sefirot::field::FieldId;

use super::prelude::*;
pub use crate::prelude::*;

fn compute_kernel(
    device: Res<Device>,
    world: Res<World>,
    mut parameters: ResMut<DebugParameters>,
    render: Res<RenderFields>,
) {
    if parameters.current_field == parameters.active_field {
        return;
    }
    parameters.kernel = Kernel::<fn()>::build(
        &device,
        &**world,
        &track!(|cell| {
            let field = parameters.active_field;
            let color = if let Some(field) = field.get_typed::<Expr<bool>, Cell>() {
                if field.expr(&cell) {
                    Vec3::splat_expr(1.0_f32)
                } else {
                    Vec3::splat_expr(0.0_f32)
                }
            } else if let Some(field) = field.get_typed::<Expr<f32>, Cell>() {
                Vec3::splat(1.0) * field.expr(&cell)
            } else if let Some(field) = field.get_typed::<Expr<Vec3<f32>>, Cell>() {
                field.expr(&cell)
            } else if let Some(field) = field.get_typed::<Expr<Vec2<i32>>, Cell>() {
                Vec3::splat(1.0) * field.expr(&cell).cast_f32().norm() / 8.0
            } else {
                panic!("Invalid field type");
            };
            *render.color.var(&cell) = color;
        }),
    )
    .with_name("debug_color");
    parameters.current_field = parameters.active_field;
}

fn color(parameters: Res<DebugParameters>) -> impl AsNodes {
    parameters.running.then(|| parameters.kernel.dispatch())
}

#[derive(Resource, Debug)]
pub struct DebugParameters {
    pub running: bool,
    pub active_field: FieldId,
    current_field: FieldId,

    kernel: Kernel<fn()>,
}
impl FromWorld for DebugParameters {
    fn from_world(world: &mut BevyWorld) -> Self {
        let empty_field = FieldId::unique();
        Self {
            running: true,
            active_field: empty_field,
            current_field: empty_field,
            kernel: Kernel::null(world.resource::<Device>()),
        }
    }
}

pub struct DebugPlugin;
impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DebugParameters>().add_systems(
            Render,
            (compute_kernel, add_render(color))
                .chain()
                .in_set(RenderPhase::Light),
        );
    }
}
