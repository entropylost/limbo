use std::f32::consts::TAU;

use super::direction::Direction;
use super::physics::NULL_OBJECT;
use crate::prelude::*;
use crate::utils::rand_f32;
use crate::world::physics::PhysicsFields;

const OUTFLOW_SIZE: f32 = 0.1;
const CELL_OUT: f32 = 0.5 + OUTFLOW_SIZE;
const MAX_VEL: f32 = 1.0 - OUTFLOW_SIZE;

#[derive(Resource)]
pub struct ImfFields {
    pub mass: VField<f32, Vec2<i32>>,
    pub next_mass: VField<f32, Vec2<i32>>,
    pub velocity: VField<Vec2<f32>, Vec2<i32>>,
    pub next_velocity: VField<Vec2<f32>, Vec2<i32>>,
    _fields: FieldSet,
}

fn setup_imf(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let imf = ImfFields {
        mass: *fields.create_bind("imf-mass", world.create_buffer(&device)),
        next_mass: *fields.create_bind("imf-next-mass", world.create_buffer(&device)),
        velocity: fields.create_bind("imf-velocity", world.create_texture(&device)),
        next_velocity: fields.create_bind("imf-next-velocity", world.create_texture(&device)),
        _fields: fields,
    };
    commands.insert_resource(imf);
}

#[kernel]
fn divergence_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    const MAX_PRESSURE: f32 = 6.0;
    Kernel::build(&device, &world.margolus(), &|el| {
        let pressure = f32::var_zeroed();
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = el.at(*el + offset);
            *pressure += imf.next_mass.expr(&oel);
        }
        let pressure_force = luisa::max(pressure - MAX_PRESSURE, 0.0);
        for dir in Direction::iter_diag() {
            let offset = dir.as_vector().map(|x| x.max(0));
            let offset = Vec2::from(offset);
            let oel = el.at(*el + offset);
            *imf.next_velocity.var(&oel) += 0.1 * dir.as_vec().expr().cast_f32() * pressure_force;
        }
    })
}

#[kernel]
fn copy_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        *imf.mass.var(&el) = imf.next_mass.expr(&el) * 0.99;
        *imf.velocity.var(&el) = (imf.next_velocity.expr(&el)).clamp(-MAX_VEL, MAX_VEL);
    })
}

#[kernel]
fn advect_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let momentum = Vec2::<f32>::var_zeroed();
        let mass = f32::var_zeroed();
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
                *mass += transferred_mass;
                *momentum += transferred_mass * vel;
            }
        }
        if mass > 0.001 {
            *imf.next_mass.var(&el) = mass;
            *imf.next_velocity.var(&el) = momentum / mass;
        } else {
            *imf.next_mass.var(&el) = mass;
            *imf.next_velocity.var(&el) = Vec2::expr(0.0, 0.0);
        }
    })
}

//  #[kernel(run)]
//  fn load_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
//      Kernel::build(&device, &**world, &|el| {
//          if (el.cast_f32() - Vec2::new(64.0, 80.0)).norm() < 30.0 {
//              *imf.mass.var(&el) += 0.3;
//              // *imf.velocity.var(&el) = Vec2::expr(0.0, 0.0);
//          }
//      })
//  }

#[kernel]
fn collide_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        if physics.object.expr(&el) != NULL_OBJECT {
            *imf.mass.var(&el) += 0.2;
            // *imf.velocity.var(&el).y = 1.0;
            // Vec2::expr(
            //     rand_f32((*el + 64_i32).cast_u32(), t, 0) - 0.5,
            //     rand_f32((*el + 64_i32).cast_u32(), t, 1) - 0.5,
            // )
            // .normalize();
        }
    })
}

fn update_imf() -> impl AsNodes {
    (
        collide_kernel.dispatch(),
        advect_kernel.dispatch(),
        divergence_kernel.dispatch(),
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
                    // init_load_kernel,
                    init_copy_kernel,
                    init_collide_kernel,
                    init_divergence_kernel,
                ),
            )
            // .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_imf).in_set(UpdatePhase::Step),
            );
    }
}
