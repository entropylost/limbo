use std::f32::consts::TAU;

use super::direction::Direction;
use super::physics::NULL_OBJECT;
use crate::prelude::*;
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
    pub value: VField<[f32; 9], Vec2<i32>>,
    pub next_value: VField<[f32; 9], Vec2<i32>>,
    pub pressure: VField<f32, Vec2<i32>>,
    pub velocity: VField<Vec2<f32>, Vec2<i32>>,
    _fields: FieldSet,
}

fn setup_imf(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let imf = ImfFields {
        value: *fields.create_bind("imf-value", world.create_buffer(&device)),
        next_value: *fields.create_bind("imf-value", world.create_buffer(&device)),
        pressure: fields.create_bind("imf-pressure", world.create_texture(&device)),
        velocity: fields.create_bind("imf-velocity", world.create_texture(&device)),
        _fields: fields,
    };
    commands.insert_resource(imf);
}

#[kernel]
fn compute_moments_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let pressure = f32::var_zeroed();
        let momentum = Vec2::<f32>::var_zeroed();
        for dir in Direction::iter_all() {
            //let weight = lattice_weight(dir);
            let index = dir.expr().as_u8().as_u32();
            let value = imf.value.expr(&el).read(index); //* weight;
            *pressure += value;
            *momentum += value * Vec2::from(dir.as_vector().cast::<f32>());
        }
        *imf.pressure.var(&el) = pressure;
        let velocity = if pressure > 0.01 {
            momentum / pressure
        } else {
            Vec2::splat_expr(0.0)
        };
        *imf.velocity.var(&el) = velocity;
    })
}

#[kernel]
fn stream_kernel(device: Res<Device>, world: Res<World>, imf: Res<ImfFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let data = imf.value.var(&el);
        for dir in Direction::iter_all() {
            let index = dir as u32;
            let lookup = el.at(*el - dir.as_vec());
            if world.contains(&lookup) {
                data.write(index, imf.next_value.expr(&lookup).read(index));
            } else {
                data.write(index, 0.0);
            }
        }
    })
}

#[kernel]
fn collision_kernel(
    device: Res<Device>,
    world: Res<World>,
    imf: Res<ImfFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn(f32)> {
    Kernel::build(&device, &**world, &|el, relaxation| {
        let cs = 10.0_f32;

        let value = imf.value.expr(&el).var();
        let pressure = imf.pressure.expr(&el);
        let velocity = imf.velocity.expr(&el);
        for dir in Direction::iter_all() {
            let dp = dir.as_vec().expr().cast_f32().dot(velocity);
            let equilibrium = pressure
                * lattice_weight(dir)
                * (1.0 + dp / (cs * cs) + (dp * dp) / (cs * cs * cs * cs)
                    - velocity.norm_squared() / (2.0 * cs * cs));
            let external_force = if physics.object.expr(&el) != NULL_OBJECT {
                Vec2::expr(10.0, 0.0)
            } else {
                Vec2::expr(0.0, -0.1)
            };

            let force = (1.0 - 1.0 / (2.0 * relaxation))
                * lattice_weight(dir)
                * ((dir.as_vec().expr().cast_f32() - velocity) / (cs * cs)
                    + dp / (cs * cs * cs * cs) * dir.as_vec().expr().cast_f32())
                .dot(external_force);
            let index = dir as u32;
            let data = value.read(index);
            value.write(index, data + (equilibrium - data) / relaxation + force);
        }
        *imf.next_value.var(&el) = value;
    })
}

#[kernel(run)]
fn load_player_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let data = imf.value.var(&el);
        for dir in Direction::iter_all() {
            let index = dir as u32;
            data.write(index, lattice_weight(dir) * 1.0);
        }
    })
}

fn update_imf() -> impl AsNodes {
    (
        // load_player_kernel.dispatch(),
        compute_moments_kernel.dispatch(),
        collision_kernel.dispatch(&10.0),
        stream_kernel.dispatch(),
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
                    init_stream_kernel,
                    init_load_player_kernel,
                    init_compute_moments_kernel,
                    init_collision_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(load_player))
            .add_systems(
                WorldUpdate,
                add_update(update_imf).in_set(UpdatePhase::Step),
            );
    }
}
