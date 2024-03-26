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

fn setup_imf(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let imf = ImpellerFields {
        divergence: fields.create_bind("imf-divergence", world.create_texture(&device)),
        edgevel: fields.create_bind("imf-edgevel", world.dual.create_texture(&device)),
        accel: fields.create_bind("imf-accel", world.create_texture(&device)),
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
    imf: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.checkerboard(), &|cell| {
        let divergence = f32::var_zeroed();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *divergence += imf.edgevel.expr(&edge) * dir.signf();
        }
        let expected_divergence = imf.divergence.expr(&cell);
        let delta = (expected_divergence - divergence) / 4.0;
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *imf.edgevel.var(&edge) += delta * dir.signf();
        }
    })
}

#[kernel]
fn accel_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImpellerFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let accel = Vec2::<f32>::var_zeroed();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *accel += imf.edgevel.expr(&edge) * dir.as_vec_f32() * dir.signf();
        }
        *imf.accel.var(&cell) = accel;
    })
}

#[kernel]
fn pressure_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImpellerFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.margolus(), &|cell| {
        // const MAX_PRESSURE: f32 = 6.0;
        let pressure = f32::var_zeroed();
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = cell.at(*cell + offset);
            *pressure += imf.next_mass.expr(&oel);
        }
        let pressure_force = 0.05 * pressure;
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = cell.at(*cell + offset);
            *imf.next_velocity.var(&oel) += dir.as_vec_f32() * pressure_force;
        }
    })
}

#[kernel]
fn copy_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImpellerFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *imf.mass.var(&cell) = imf.next_mass.expr(&cell) * 0.99;
        *imf.velocity.var(&cell) = imf.next_velocity.expr(&cell) + 0.01 * imf.accel.expr(&cell);
        let norm = imf.velocity.expr(&cell).norm();
        if norm > MAX_VEL {
            *imf.velocity.var(&cell) *= MAX_VEL / norm;
        }
        *imf.object.var(&cell) = imf.next_object.expr(&cell);
    })
}

#[kernel]
fn advect_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImpellerFields>) -> Kernel<fn()> {
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

        *imf.next_mass.var(&cell) = mass;
        *imf.next_velocity.var(&cell) = if mass > 0.0001 {
            momentum / mass
        } else {
            Vec2::expr(0.0, 0.0)
        };
        *imf.next_object.var(&cell) = objects.read(max_index);
    })
}

#[kernel(run)]
fn load_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImpellerFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *imf.object.var(&cell) = NULL_OBJECT;
    })
}

#[kernel]
fn collide_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImpellerFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if physics.object.expr(&cell) == 1 || physics.object.expr(&cell) == 2 {
            let last_mass = imf.mass.expr(&cell);
            *imf.mass.var(&cell) += 0.1;
            *imf.object.var(&cell) = physics.object.expr(&cell);
            *imf.velocity.var(&cell) = ((imf.velocity.var(&cell) * last_mass
                + 0.1 * physics.velocity.expr(&cell))
                / imf.mass.expr(&cell))
            .clamp(-MAX_VEL, MAX_VEL);
        }
        if physics.object.expr(&cell) == 1 || physics.object.expr(&cell) == 2 {
            *imf.divergence.var(&cell) = 1.0;
        } else if physics.object.expr(&cell) == 0 {
            *imf.divergence.var(&cell) = -3.0;
        } else {
            *imf.divergence.var(&cell) = 0.0;
        }
    })
}

pub fn update_imf() -> impl AsNodes {
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

pub struct ImfPlugin;
impl Plugin for ImfPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_imf)
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
                add_update(update_imf).in_set(UpdatePhase::Step),
            );
    }
}
