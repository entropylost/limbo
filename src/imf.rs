use crate::physics::PhysicsFields;
use crate::prelude::*;

pub const IMF_CAP: u32 = 2048;

#[derive(Resource)]
pub struct ImfFields {
    pub value: AField<u32, Vec2<i32>>,
    pub next_value: AField<u32, Vec2<i32>>,
    pub out: VField<Vec2<i32>, Vec2<i32>>,
    pub valid: VField<bool, Vec2<i32>>,
    _fields: FieldSet,
}

fn setup_imf(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let imf = ImfFields {
        value: fields.create_bind("imf-value", world.create_buffer_morton(&device)),
        next_value: fields.create_bind("imf-value", world.create_buffer_morton(&device)),
        out: fields.create_bind("imf-out", world.create_texture(&device)),
        valid: *fields.create_bind("imf-valid", world.create_buffer_morton(&device)),
        _fields: fields,
    };
    commands.insert_resource(imf);
}

#[kernel]
fn update_valid(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        *imf.valid.var(&el) = &imf.value.expr(&el.at(imf.out.expr(&el))) < IMF_CAP / 2;
    })
}

#[kernel]
fn init_imf_out(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        *imf.out.var(&el) = *el;
    })
}

#[kernel]
fn propegate_imf_out(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let best_dist = i32::MAX.var();
        let best_out = (*el).var();
        let pos = *el;
        world.on_adjacent(&el, |el| {
            if imf.valid.expr(&el) {
                let out = imf.out.expr(&el);
                let delta = out - pos;
                let dist = delta.x * delta.x + delta.y * delta.y;
                if dist < best_dist {
                    *best_dist = dist;
                    *best_out = out;
                }
            }
        });
        // TODO: Move up to defn to simplify.
        let out = imf.out.expr(&el);
        if imf.valid.expr(&el) {
            let delta = out - pos;
            let dist = delta.x * delta.x + delta.y * delta.y;
            if dist < best_dist {
                *best_dist = dist;
                *best_out = out;
            }
        }
        if imf.value.expr(&el) < IMF_CAP / 2 {
            *best_dist = 0;
            *best_out = pos;
        }
        // TODO: Also check the current out to see if it's also good?
        if best_dist < i32::MAX {
            *imf.out.var(&el) = best_out;
        }
    })
}

#[kernel]
fn imf_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let object = physics.object.expr(&el);
        let value = imf.value.expr(&el);
        let next_value = imf.next_value.atomic(&el);
        if object == 1 {
            // Player
            next_value.fetch_add(IMF_CAP / 16);
        };
        if value > IMF_CAP && imf.valid.expr(&el) {
            let diff = value - IMF_CAP;
            next_value.fetch_sub(diff);
            let out = el.at(imf.out.expr(&el));
            imf.next_value.atomic(&out).fetch_add(diff);
        }
        if value >= 1 {
            next_value.fetch_sub(1);
        }
    })
}

#[kernel]
fn copy_next_imf_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        *imf.value.var(&el) = imf.next_value.expr(&el);
    })
}

fn init_imf() -> impl AsNodes {
    (init_imf_out.dispatch(), update_valid.dispatch())
}

fn update_imf() -> impl AsNodes {
    (
        propegate_imf_out.dispatch(),
        update_valid.dispatch(),
        imf_kernel.dispatch(),
        copy_next_imf_kernel.dispatch(),
    )
}

pub struct ImfPlugin;
impl Plugin for ImfPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_imf)
            .add_systems(
                InitKernel,
                (
                    init_init_imf_out,
                    init_update_valid,
                    init_propegate_imf_out,
                    init_imf_kernel,
                    init_copy_next_imf_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(init_imf))
            .add_systems(
                WorldUpdate,
                add_update(update_imf).in_set(UpdatePhase::Step),
            );
    }
}
