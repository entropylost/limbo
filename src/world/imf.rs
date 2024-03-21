use super::direction::Direction;
use super::physics::NULL_OBJECT;
use crate::prelude::*;
use crate::world::physics::PhysicsFields;

const OUTFLOW_SIZE: f32 = 0.1;
const CELL_OUT: f32 = 0.5 + OUTFLOW_SIZE;
const MAX_VEL: f32 = 1.0 - OUTFLOW_SIZE;

#[derive(Resource)]
pub struct ImfFields {
    pub mass: VField<f32, Cell>,
    pub next_mass: VField<f32, Cell>,
    pub velocity: VField<Vec2<f32>, Cell>,
    pub next_velocity: VField<Vec2<f32>, Cell>,
    pub object: VField<u32, Cell>,
    pub next_object: VField<u32, Cell>,
    _fields: FieldSet,
}

fn setup_imf(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let imf = ImfFields {
        mass: *fields.create_bind("imf-mass", world.create_buffer(&device)),
        next_mass: *fields.create_bind("imf-next-mass", world.create_buffer(&device)),
        velocity: fields.create_bind("imf-velocity", world.create_texture(&device)),
        next_velocity: fields.create_bind("imf-next-velocity", world.create_texture(&device)),
        object: fields.create_bind("imf-object", world.create_texture(&device)),
        next_object: fields.create_bind("imf-next-object", world.create_texture(&device)),
        _fields: fields,
    };
    commands.insert_resource(imf);
}

#[kernel]
fn divergence_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
) -> Kernel<fn(u32)> {
    Kernel::build(&device, &world.margolus(), &|el, i| {
        use luisa::lang::ops::RemEuclidExpr;
        let i = i.cast_i32() % 4;
        if el.x.rem_euclid(2) != i / 2 || el.y.rem_euclid(2) != i % 2 {
            return;
        }

        // const MAX_PRESSURE: f32 = 6.0;
        let pressure = f32::var_zeroed();
        let divergence = f32::var_zeroed();
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = el.at(*el + offset);
            *pressure += imf.next_mass.expr(&oel);
            // TODO: This should use the velocity after advection.
            *divergence +=
                imf.next_velocity.expr(&oel).dot(dir.as_vec_f32()) * imf.next_mass.expr(&oel);
        }
        let pressure_force = (pressure - 0.5) * 0.002 - 0.25 * 0.8_f32 * divergence;
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = el.at(*el + offset);
            *imf.next_velocity.var(&oel) +=
                dir.as_vec_f32() * pressure_force / (imf.next_mass.expr(&oel) + 0.00001);
        }
    })
}

#[kernel]
fn copy_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        *imf.mass.var(&el) = imf.next_mass.expr(&el) * 0.99;
        *imf.velocity.var(&el) = (imf.next_velocity.expr(&el)).clamp(-MAX_VEL, MAX_VEL);
        *imf.object.var(&el) = imf.next_object.expr(&el);
    })
}

#[kernel]
fn advect_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        // TODO: Track highest index.
        let objects = [NULL_OBJECT; 9].var();
        let masses = [0.0_f32; 9].var();
        let momenta = [Vec2::splat(0.0_f32); 9].var();

        for dx in -1..=1 {
            for dy in -1..=1 {
                let pos = el.at(Vec2::expr(dx, dy) + *el);
                if !world.contains(&pos) {
                    continue;
                }
                let vel = imf.velocity.expr(&pos);
                let offset = vel + Vec2::<i32>::expr(dx, dy).cast_f32();
                let intersect = luisa::max(
                    luisa::min(
                        luisa::min(offset + 0.5 + CELL_OUT, 0.5 + CELL_OUT - offset),
                        1.0,
                    ) / (CELL_OUT * 2.0),
                    0.0,
                );
                let weight = intersect.x * intersect.y;
                let transferred_mass = imf.mass.expr(&pos) * weight;
                let object = imf.object.expr(&pos);
                for i in 0_u32..9_u32 {
                    if objects.read(i) == object {
                        masses.write(i, masses.read(i) + transferred_mass);
                        momenta.write(i, momenta.read(i) + vel * transferred_mass);
                        break;
                    } else if objects.read(i) == NULL_OBJECT {
                        objects.write(i, object);
                        masses.write(i, masses.read(i) + transferred_mass);
                        momenta.write(i, momenta.read(i) + vel * transferred_mass);
                        break;
                    }
                }
            }
        }

        let max_index = 0_u32.var();
        let max_mass = f32::var_zeroed();
        let mass_sum = f32::var_zeroed();
        let momentum_sum = Vec2::<f32>::var_zeroed();

        for i in 0_u32..9 {
            if masses.read(i) >= max_mass {
                *max_mass = masses.read(i);
                *max_index = i;
            }
            *mass_sum += masses.read(i);
            *momentum_sum += momenta.read(i);
        }

        let mass = luisa::max(max_mass * 2.0 - mass_sum, 0.0);
        let momentum = momenta[max_index] * 2.0 - momentum_sum;

        if mass > 0.0001 {
            *imf.next_mass.var(&el) = mass;
            *imf.next_velocity.var(&el) = momentum / mass;
            *imf.next_object.var(&el) = objects.read(max_index);
        } else {
            *imf.next_mass.var(&el) = mass;
            *imf.next_velocity.var(&el) = Vec2::expr(0.0, 0.0);
        }
    })
}

#[kernel(run)]
fn load_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        *imf.object.var(&el) = NULL_OBJECT;
    })
}

#[kernel]
fn collide_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        if physics.object.expr(&el) == 1 {
            let last_mass = imf.mass.expr(&el);
            *imf.mass.var(&el) += 0.1;
            *imf.object.var(&el) = physics.object.expr(&el);
            *imf.velocity.var(&el) = ((imf.velocity.var(&el) * last_mass
                + 0.1 * physics.velocity.expr(&el))
                / imf.mass.expr(&el))
            .clamp(-MAX_VEL, MAX_VEL);
        }
    })
}

#[kernel]
fn collide_null_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        if physics.object.expr(&el) == 0 {
            *imf.next_mass.var(&el) = 0.0;
            // *imf.next_velocity.var(&el) = Vec2::expr(0.0, 0.4);
        }
    })
}

pub fn update_imf(mut i: Local<u32>) -> impl AsNodes {
    *i += 1;
    (
        collide_kernel.dispatch(),
        advect_kernel.dispatch(),
        collide_null_kernel.dispatch(),
        divergence_kernel.dispatch(&*i),
        copy_kernel.dispatch(),
    )
        .chain()
}

pub struct ImfPlugin;
impl Plugin for ImfPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_imf)
            .add_systems(
                InitKernel,
                (
                    init_advect_kernel,
                    init_load_kernel,
                    init_copy_kernel,
                    init_collide_kernel,
                    init_collide_null_kernel,
                    init_divergence_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_imf).in_set(UpdatePhase::Step),
            );
    }
}
