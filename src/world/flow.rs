use super::imf::{update_imf, ImfFields};
use crate::prelude::*;
use crate::utils::rand_f32;

#[derive(Resource)]
pub struct FlowFields {
    pub activation: VField<bool, Cell>,
    _fields: FieldSet,
}

fn setup_flow(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let flow = FlowFields {
        activation: *fields.create_bind("flow-activation", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(flow);
}

#[kernel]
fn flow_update_kernel(
    device: Res<Device>,
    world: Res<World>,
    flow: Res<FlowFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn(u32)> {
    Kernel::build(&device, &**world, &|cell, t| {
        if flow.activation.expr(&cell) {
            let vel = imf.velocity.expr(&cell);
            let sign = vel.signum().cast_i32();
            let abs = vel.abs();
            let int = abs.floor();
            let frac = abs - int;
            let abs = (Vec2::expr(
                rand_f32(dispatch_id().xy(), t, 1),
                rand_f32(dispatch_id().xy(), t, 2),
            ) < frac)
                .cast_i32()
                + int.cast_i32();
            let pos = *cell + sign * abs;
            if !world.contains(&pos) {
                return;
            }
            if (pos != *cell).any() {
                *flow.activation.var(&cell.at(pos)) = true;
                *flow.activation.var(&cell) = false;
            }
        } else if rand_f32(dispatch_id().xy(), t, 0) < 0.0001 {
            *flow.activation.var(&cell) = true;
        }
    })
}

fn flow_update(mut t: Local<u32>) -> impl AsNodes {
    *t += 1;
    flow_update_kernel.dispatch(&*t)
}

pub struct FlowPlugin;
impl Plugin for FlowPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_flow)
            .add_systems(InitKernel, init_flow_update_kernel)
            .add_systems(
                WorldUpdate,
                add_update(flow_update)
                    .in_set(UpdatePhase::Step)
                    .after(update_imf),
            );
    }
}
