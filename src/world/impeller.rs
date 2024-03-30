use super::direction::Direction;
use super::physics::NULL_OBJECT;
use crate::prelude::*;
use crate::world::physics::PhysicsFields;

// TODO: Make the blur have less artifacting in orthogonal directions.
const OUTFLOW_SIZE: f32 = 0.1;
const CELL_OUT: f32 = 0.5 + OUTFLOW_SIZE;
const MAX_VEL: f32 = 1.0 - OUTFLOW_SIZE;

#[derive(Resource)]
pub struct ImpellerFields {
    pub divergence: VField<f32, Cell>,
    pub edgevel: VField<f32, Edge>,
    pub accel: VField<Vec2<f32>, Cell>,
    pub mass: VField<f32, Cell>,
    pub next_mass: VField<f32, Cell>,
    pub velocity: VField<Vec2<f32>, Cell>,
    pub next_velocity: VField<Vec2<f32>, Cell>,
    pub object: VField<u32, Cell>,
    pub next_object: VField<u32, Cell>,
    _fields: FieldSet,
}

fn setup_impeller(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let impeller = ImpellerFields {
        divergence: fields.create_bind("impeller-divergence", world.create_texture(&device)),
        edgevel: fields.create_bind("impeller-edgevel", world.dual.create_texture(&device)),
        accel: fields.create_bind("impeller-accel", world.create_texture(&device)),
        mass: *fields.create_bind("impeller-mass", world.create_buffer(&device)),
        next_mass: *fields.create_bind("impeller-next-mass", world.create_buffer(&device)),
        velocity: fields.create_bind("impeller-velocity", world.create_texture(&device)),
        next_velocity: fields.create_bind("impeller-next-velocity", world.create_texture(&device)),
        object: fields.create_bind("impeller-object", world.create_texture(&device)),
        next_object: fields.create_bind("impeller-next-object", world.create_texture(&device)),
        _fields: fields,
    };
    commands.insert_resource(impeller);
}

#[kernel]
fn divergence_kernel(
    device: Res<Device>,
    world: Res<World>,
    impeller: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.checkerboard(), &|cell| {
        let divergence = f32::var_zeroed();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *divergence += impeller.edgevel.expr(&edge) * dir.signf();
        }
        let expected_divergence = impeller.divergence.expr(&cell);
        let delta = (expected_divergence - divergence) / 4.0;
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *impeller.edgevel.var(&edge) += delta * dir.signf();
        }
    })
}

#[kernel]
fn accel_kernel(
    device: Res<Device>,
    world: Res<World>,
    impeller: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let accel = Vec2::<f32>::var_zeroed();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *accel += impeller.edgevel.expr(&edge) * dir.as_vec_f32() * dir.signf();
        }
        *impeller.accel.var(&cell) = accel;
    })
}

#[kernel]
fn pressure_kernel(
    device: Res<Device>,
    world: Res<World>,
    impeller: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.margolus(), &|cell| {
        // const MAX_PRESSURE: f32 = 6.0;
        let pressure = f32::var_zeroed();
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = cell.at(*cell + offset);
            *pressure += impeller.next_mass.expr(&oel);
        }
        let pressure_force = 0.05 * pressure;
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = cell.at(*cell + offset);
            *impeller.next_velocity.var(&oel) += dir.as_vec_f32() * pressure_force;
        }
    })
}

#[kernel]
fn copy_kernel(
    device: Res<Device>,
    world: Res<World>,
    impeller: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *impeller.mass.var(&cell) = impeller.next_mass.expr(&cell) * 0.99;
        *impeller.velocity.var(&cell) =
            impeller.next_velocity.expr(&cell) + 0.01 * impeller.accel.expr(&cell);
        let norm = impeller.velocity.expr(&cell).norm();
        if norm > MAX_VEL {
            *impeller.velocity.var(&cell) *= MAX_VEL / norm;
        }
        *impeller.object.var(&cell) = impeller.next_object.expr(&cell);
    })
}

#[kernel]
fn advect_kernel(
    device: Res<Device>,
    world: Res<World>,
    impeller: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let objects = [NULL_OBJECT; 9].var();
        let masses = [0.0_f32; 9].var();
        let momenta = [Vec2::splat(0.0_f32); 9].var();

        for dx in -1..=1 {
            for dy in -1..=1 {
                let pos = cell.at(Vec2::expr(dx, dy) + *cell);
                if !world.contains(&pos) {
                    continue;
                }
                let vel = impeller.velocity.expr(&pos);
                let offset = vel + Vec2::<i32>::expr(dx, dy).cast_f32();
                let intersect = luisa::max(
                    luisa::min(
                        luisa::min(offset + 0.5 + CELL_OUT, 0.5 + CELL_OUT - offset),
                        1.0,
                    ) / (CELL_OUT * 2.0),
                    0.0,
                );
                let weight = intersect.x * intersect.y;
                let transferred_mass = impeller.mass.expr(&pos) * weight;
                let object = impeller.object.expr(&pos);
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

        *impeller.next_mass.var(&cell) = mass;
        *impeller.next_velocity.var(&cell) = if mass > 0.0001 {
            momentum / mass
        } else {
            Vec2::expr(0.0, 0.0)
        };
        *impeller.next_object.var(&cell) = objects.read(max_index);
    })
}

#[kernel(run)]
fn load_kernel(
    device: Res<Device>,
    world: Res<World>,
    impeller: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *impeller.object.var(&cell) = NULL_OBJECT;
    })
}

#[kernel]
fn collide_kernel(
    device: Res<Device>,
    world: Res<World>,
    impeller: Res<ImpellerFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if physics.object.expr(&cell) == 1 || physics.object.expr(&cell) == 2 {
            let last_mass = impeller.mass.expr(&cell);
            *impeller.mass.var(&cell) += 0.1;
            *impeller.object.var(&cell) = physics.object.expr(&cell);
            *impeller.velocity.var(&cell) = ((impeller.velocity.var(&cell) * last_mass
        /* + 0.1 * physics.velocity.expr(&cell) */)
                / impeller.mass.expr(&cell))
            .clamp(-MAX_VEL, MAX_VEL);
        }
        if physics.object.expr(&cell) == 1 || physics.object.expr(&cell) == 2 {
            *impeller.divergence.var(&cell) = 1.0;
        } else if physics.object.expr(&cell) == 0 {
            *impeller.divergence.var(&cell) = -3.0;
        } else {
            *impeller.divergence.var(&cell) = 0.0;
        }
    })
}

pub fn update_impeller() -> impl AsNodes {
    (
        collide_kernel.dispatch(),
        divergence_kernel.dispatch(),
        accel_kernel.dispatch(),
        advect_kernel.dispatch(),
        pressure_kernel.dispatch(),
        copy_kernel.dispatch(),
    )
        .chain()
}

pub struct ImpellerPlugin;
impl Plugin for ImpellerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_impeller)
            .add_systems(
                InitKernel,
                (
                    init_divergence_kernel,
                    init_accel_kernel,
                    init_advect_kernel,
                    init_load_kernel,
                    init_copy_kernel,
                    init_collide_kernel,
                    init_pressure_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_impeller).in_set(UpdatePhase::Step),
            );
    }
}
