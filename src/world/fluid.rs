use super::direction::Direction;
use crate::prelude::*;

const OUTFLOW_SIZE: f32 = 0.0;
const CELL_OUT: f32 = 0.5 + OUTFLOW_SIZE;
const MAX_VEL: f32 = 1.0 - OUTFLOW_SIZE;

#[derive(Resource)]
pub struct FluidFields {
    pub mass: VField<f32, Cell>,
    pub next_mass: VField<f32, Cell>,
    pub velocity: VField<Vec2<f32>, Cell>,
    pub next_velocity: VField<Vec2<f32>, Cell>,
    pub solid: VField<bool, Cell>,
    _fields: FieldSet,
}

fn setup_fluids(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let fluid = FluidFields {
        mass: fields.create_bind("fluid-mass", world.create_texture(&device)),
        next_mass: fields.create_bind("fluid-next-mass", world.create_texture(&device)),
        velocity: fields.create_bind("fluid-velocity", world.create_texture(&device)),
        next_velocity: fields.create_bind("fluid-next-velocity", world.create_texture(&device)),
        solid: *fields.create_bind("fluid-solid", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(fluid);
}

#[kernel]
fn pressure_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.margolus(), &|cell| {
        const MAX_PRESSURE: f32 = 4.0;
        let pressure = f32::var_zeroed();
        let divergence = f32::var_zeroed();

        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = cell.at(*cell + offset);
            *pressure += fluid.next_mass.expr(&oel);
            *divergence +=
                fluid.next_velocity.expr(&oel).dot(dir.as_vec_f32()) * fluid.next_mass.expr(&oel);
        }
        let pressure_force = 0.003 * max(pressure - MAX_PRESSURE, 0.0) - 0.01 * divergence;
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = cell.at(*cell + offset);
            *fluid.next_velocity.var(&oel) +=
                dir.as_vec_f32() * pressure_force / (fluid.next_mass.expr(&oel) + 0.01);
        }
    })
}

#[kernel]
fn copy_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *fluid.mass.var(&cell) = fluid.next_mass.expr(&cell);
        *fluid.velocity.var(&cell) =
            fluid.next_velocity.expr(&cell) * 0.999 + Vec2::expr(0.0, -0.001);
        let norm = fluid.velocity.expr(&cell).norm();
        if norm > MAX_VEL {
            *fluid.velocity.var(&cell) *= MAX_VEL / norm;
        }
        if fluid.solid.expr(&cell) {
            *fluid.velocity.var(&cell) = Vec2::expr(0.0, 0.0);
        }
    })
}

#[kernel]
fn advect_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let momentum = Vec2::<f32>::var_zeroed();
        let mass = f32::var_zeroed();
        for dx in -1..=1 {
            for dy in -1..=1 {
                let pos = cell.at(Vec2::expr(dx, dy) + *cell);
                if !world.contains(&pos) {
                    continue;
                }
                let vel = fluid.velocity.expr(&pos);
                let offset = vel + Vec2::<i32>::expr(dx, dy).cast_f32();
                let intersect = luisa::max(
                    luisa::min(
                        luisa::min(offset + 0.5 + CELL_OUT, 0.5 + CELL_OUT - offset),
                        1.0,
                    ) / (CELL_OUT * 2.0),
                    0.0,
                );
                let weight = intersect.x * intersect.y;
                let transferred_mass = fluid.mass.expr(&pos) * weight;
                *mass += transferred_mass;
                *momentum += transferred_mass * vel;
            }
        }
        if mass > 0.001 {
            *fluid.next_mass.var(&cell) = mass;
            *fluid.next_velocity.var(&cell) = momentum / mass;
        } else {
            *fluid.next_mass.var(&cell) = mass;
            *fluid.next_velocity.var(&cell) = Vec2::expr(0.0, 0.0);
        }
    })
}

#[kernel(run)]
fn load_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if (*cell - 64).cast_f32().norm() < 32.0 {
            *fluid.mass.var(&cell) = 1.0;
        } else {
            *fluid.mass.var(&cell) = 0.0;
            *fluid.velocity.var(&cell) = Vec2::expr(0.0, 0.0);
        }
        if cell.y < 20 || cell.x < 20 || cell.x > 256 - 20 {
            *fluid.solid.var(&cell) = true;
            *fluid.mass.var(&cell) = 3.0;
        } else {
            *fluid.solid.var(&cell) = false;
        }
    })
}

fn update_fluids(mut index: Local<u8>) -> impl AsNodes {
    *index = index.wrapping_add(1);
    (
        advect_kernel.dispatch(),
        pressure_kernel.dispatch(),
        pressure_kernel.dispatch(),
        pressure_kernel.dispatch(),
        pressure_kernel.dispatch(),
        pressure_kernel.dispatch(),
        pressure_kernel.dispatch(),
        pressure_kernel.dispatch(),
        copy_kernel.dispatch(),
    )
        .chain()
}

pub struct FluidPlugin;
impl Plugin for FluidPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_fluids)
            .add_systems(
                InitKernel,
                (
                    init_advect_kernel,
                    init_copy_kernel,
                    init_pressure_kernel,
                    init_load_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_fluids).in_set(UpdatePhase::Step),
            );
    }
}
