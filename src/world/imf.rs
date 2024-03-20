use std::f32::consts::TAU;

use super::direction::Direction;
use super::physics::NULL_OBJECT;
use crate::prelude::*;
use crate::utils::rand_f32;
use crate::world::physics::PhysicsFields;

#[tracked]
fn vector_weight(v: Expr<Vec2<f32>>) -> Expr<f32> {
    (-v.norm_squared() / 2.0_f32).exp() / TAU
}

fn lattice_weight(dir: Direction) -> f32 {
    if dir.ortho() {
        1.0 / 9.0
    } else if dir.diag() {
        1.0 / 36.0
    } else {
        4.0 / 9.0
    }
}

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

// #[kernel]
// fn divergence_kernel(device: Res<Device>, world: Res<World>, )

#[kernel]
fn copy_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        *imf.mass.var(&el) = imf.next_mass.expr(&el) * 0.95;
        *imf.velocity.var(&el) = imf.next_velocity.expr(&el);
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
                let intersect = luisa::max(luisa::min(offset + 1.0, 1.0 - offset), 0.0);
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

#[kernel]
fn load_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn(u32)> {
    Kernel::build(&device, &**world, &|el, t| {
        if (el.cast_f32() - Vec2::expr(64.5, 64.5)).norm() < 10.0 {
            *imf.mass.var(&el) = 5.0;
            *imf.velocity.var(&el) = (el.cast_f32() - Vec2::expr(64.5, 64.5)).normalize();
        }
        if physics.object.expr(&el) != NULL_OBJECT {
            // *imf.mass.var(&el) = 0.0;
            *imf.velocity.var(&el) = Vec2::expr(0.0, 0.0);
            Vec2::expr(
                rand_f32((*el + 64_i32).cast_u32(), t, 0) - 0.5,
                rand_f32((*el + 64_i32).cast_u32(), t, 1) - 0.5,
            )
            .normalize();
        }
    })
}

fn update_imf(mut t: Local<u32>) -> impl AsNodes {
    *t += 1;
    (
        load_kernel.dispatch(&*t),
        advect_kernel.dispatch(),
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
                (init_advect_kernel, init_copy_kernel, init_load_kernel),
            )
            // .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_imf).in_set(UpdatePhase::Step),
            );
    }
}
