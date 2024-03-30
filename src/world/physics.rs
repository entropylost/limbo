use std::f32::consts::TAU;
use std::iter::repeat;

use id_newtype::UniqueId;
use morton::deinterleave_morton;
use sefirot::domain::dynamic::DynamicDomain;
use sefirot::mapping::buffer::StaticDomain;
use sefirot::utils::Singleton;

use crate::prelude::*;

const NUM_OBJECTS: usize = 16;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, UniqueId)]
#[repr(transparent)]
pub struct ObjectHost(u32);

pub type Object = Expr<u32>;

#[repr(C)]
#[derive(Value, Debug, Copy, Clone, PartialEq, Eq)]
pub struct Collision {
    a_position: Vec2<i32>,
    predicted_collision: Vec2<i32>,
}

pub struct ObjectBuffers {
    mass: Buffer<u32>,
    moment: Buffer<u32>,
    position: Buffer<Vec2<f32>>,
    angle: Buffer<f32>,
    velocity: Buffer<Vec2<f32>>,
    angvel: Buffer<f32>,
}

#[derive(Resource)]
pub struct ObjectFields {
    // TODO: Change for resizing.
    pub domain: StaticDomain<1>,
    // Also change these to use ObjectId instead.
    pub mass: AField<u32, Object>,
    pub moment: AField<u32, Object>,
    // TODO: Need to be able to adjust these.
    // Replace with center of mass upon object breaking.
    pub position: VField<Vec2<f32>, Object>,
    pub predicted_position: VField<Vec2<f32>, Object>,
    pub angle: VField<f32, Object>,
    pub predicted_angle: VField<f32, Object>,

    pub velocity: VField<Vec2<f32>, Object>,
    pub angvel: VField<f32, Object>,
    // For collisions.
    pub impulse: AField<Vec2<f32>, Object>,
    pub angular_impulse: AField<f32, Object>,
    _fields: FieldSet,
    buffers: ObjectBuffers,
}

#[derive(Resource)]
pub struct InitData {
    pub cells: [[u32; 256]; 256],
    pub object_velocities: Vec<Vector2<f32>>,
    pub object_angvels: Vec<f32>,
}

pub const NULL_OBJECT: u32 = u32::MAX;

#[derive(Resource)]
pub struct CollisionFields {
    pub mapper: StaticDomain<1>,
    pub domain: DynamicDomain,
    pub data: VEField<Collision, u32>,
    pub next: Singleton<u32>,
    _fields: FieldSet,
}

#[derive(Resource)]
pub struct PhysicsFields {
    pub object: VField<u32, Cell>,
    pub predicted_object: VField<u32, Cell>,
    pub delta: VField<Vec2<i32>, Cell>,
    pub lock: AField<u32, Cell>,
    pub locked_revmap: VField<Vec2<i32>, Cell>,
    pub prev_rejection: VField<Vec2<i32>, Cell>,
    pub rejection: VField<Vec2<i32>, Cell>,
    _fields: FieldSet,
    object_buffer: Buffer<u32>,
    lock_buffer: Buffer<u32>,
}

fn setup_objects(mut commands: Commands, device: Res<Device>) {
    let domain = StaticDomain::<1>::new(NUM_OBJECTS as u32);

    let buffers = ObjectBuffers {
        mass: device.create_buffer(NUM_OBJECTS),
        moment: device.create_buffer(NUM_OBJECTS),
        position: device.create_buffer(NUM_OBJECTS),
        angle: device.create_buffer(NUM_OBJECTS),
        velocity: device.create_buffer(NUM_OBJECTS),
        angvel: device.create_buffer(NUM_OBJECTS),
    };

    let mut fields = FieldSet::new();

    let mass = fields.create_bind("object-mass", domain.map_buffer(buffers.mass.view(..)));
    let moment = fields.create_bind("object-moment", domain.map_buffer(buffers.moment.view(..)));

    let position = fields.create_bind(
        "object-position",
        domain.map_buffer(buffers.position.view(..)),
    );
    let predicted_position =
        fields.create_bind("object-predicted-position", domain.create_buffer(&device));
    let angle = fields.create_bind("object-angle", domain.map_buffer(buffers.angle.view(..)));
    let predicted_angle =
        fields.create_bind("object-predicted-angle", domain.create_buffer(&device));

    let velocity = fields.create_bind(
        "object-velocity",
        domain.map_buffer(buffers.velocity.view(..)),
    );
    let angvel = fields.create_bind("object-angvel", domain.map_buffer(buffers.angvel.view(..)));

    let impulse = fields.create_bind("object-impulse", domain.create_buffer(&device));
    let angular_impulse =
        fields.create_bind("object-angular-impulse", domain.create_buffer(&device));

    let objects = ObjectFields {
        domain,
        mass,
        moment,
        position,
        predicted_position,
        angle,
        predicted_angle,
        velocity,
        angvel,
        impulse,
        angular_impulse,
        _fields: fields,
        buffers,
    };
    commands.insert_resource(objects);
}

fn setup_physics(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let object_buffer = device.create_buffer((world.width() * world.height()) as usize);
    let lock_buffer = device.create_buffer((world.width() * world.height()) as usize);
    let object = *fields.create_bind("physics-object", world.map_buffer(object_buffer.view(..)));
    let predicted_object =
        *fields.create_bind("physics-predicted-object", world.create_buffer(&device));
    let delta = fields.create_bind("physics-delta", world.create_texture(&device));
    let lock = fields.create_bind("physics-lock", world.map_buffer(lock_buffer.view(..)));
    let locked_revmap = *fields.create_bind("physics-locked-revmap", world.create_buffer(&device));

    let prev_rejection = *fields.create_bind("physics-rejection", world.create_buffer(&device));
    let rejection = *fields.create_bind("physics-next-rejection", world.create_buffer(&device));

    let physics = PhysicsFields {
        object,
        predicted_object,
        delta,
        lock,
        locked_revmap,
        prev_rejection,
        rejection,
        _fields: fields,
        object_buffer,
        lock_buffer,
    };

    let mut fields = FieldSet::new();
    let mapper = StaticDomain::<1>::new(1024);
    let domain = DynamicDomain::new(0);
    let data = fields.create_bind("collision-data", mapper.create_buffer(&device));

    let collision = CollisionFields {
        mapper,
        domain,
        data,
        next: Singleton::new(&device),
        _fields: fields,
    };

    commands.insert_resource(physics);
    commands.insert_resource(collision);
}

#[tracked]
fn skew_rotate(v: Expr<Vec2<i32>>, angle: Expr<f32>) -> Expr<Vec2<i32>> {
    let a = -(angle / 2.0).tan();
    let b = angle.sin();
    let x = v.x;
    let y = v.y;
    let x = x + (y.cast_f32() * a).round().cast_i32();
    let y = y + (x.cast_f32() * b).round().cast_i32();
    let x = x + (y.cast_f32() * a).round().cast_i32();
    Vec2::expr(x, y)
}

#[tracked]
fn skew_rotate_quadrant(v: Expr<Vec2<i32>>, angle: Expr<f32>) -> Expr<Vec2<i32>> {
    let angle = angle - quadrant(angle).cast_f32() * TAU / 4.0;
    skew_rotate(v, angle)
}

#[tracked]
fn quadrant_rotate(v: Expr<Vec2<i32>>, quadrant: Expr<i32>) -> Expr<Vec2<i32>> {
    let quadrant = quadrant.rem_euclid(4);
    let v = if quadrant % 2 == 1 {
        Vec2::expr(-v.y, v.x)
    } else {
        v
    };
    if quadrant >= 2 {
        -v
    } else {
        v
    }
}

#[tracked]
fn rotate(v: Expr<Vec2<f32>>, angle: Expr<f32>) -> Expr<Vec2<f32>> {
    let x = v.x;
    let y = v.y;
    let x = x * angle.cos() - y * angle.sin();
    let y = x * angle.sin() + y * angle.cos();
    Vec2::expr(x, y)
}

#[tracked]
fn quadrant(angle: Expr<f32>) -> Expr<i32> {
    (angle * 4.0 / TAU).round().cast_i32().rem_euclid(4)
}

#[kernel]
fn clear_objects_kernel(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    world: Res<World>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *physics.object.var(&cell) = NULL_OBJECT;
    })
}

#[kernel]
fn predict_kernel(device: Res<Device>, objects: Res<ObjectFields>) -> Kernel<fn()> {
    Kernel::build(&device, &objects.domain, &|obj| {
        *objects.velocity.var(&obj) +=
            objects.impulse.expr(&obj) / objects.mass.expr(&obj).cast_f32();
        *objects.impulse.var(&obj) = Vec2::splat(0_f32);
        *objects.predicted_position.var(&obj) =
            objects.position.expr(&obj) + objects.velocity.expr(&obj);

        *objects.angvel.var(&obj) +=
            objects.angular_impulse.expr(&obj) / objects.moment.expr(&obj).cast_f32();
        *objects.angular_impulse.var(&obj) = 0.0;
        *objects.predicted_angle.var(&obj) = objects.angle.expr(&obj) + objects.angvel.expr(&obj);
    })
}

#[kernel]
fn finalize_objects_kernel(device: Res<Device>, objects: Res<ObjectFields>) -> Kernel<fn()> {
    Kernel::build(&device, &objects.domain, &|obj| {
        *objects.position.var(&obj) = objects.predicted_position.expr(&obj);
        *objects.angle.var(&obj) = objects.predicted_angle.expr(&obj);
    })
}

#[kernel]
fn finalize_move_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if physics.lock.expr(&cell) != 1 {
            *physics.object.var(&cell) = NULL_OBJECT;
        } else {
            *physics.object.var(&cell) = physics.predicted_object.expr(&cell);
        }
    })
}

#[kernel]
fn move_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
    objects: Res<ObjectFields>,
    collisions: Res<CollisionFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        // TODO: What to do about collisions?
        let obj = physics.object.expr(&cell);
        if obj == NULL_OBJECT {
            *physics.delta.var(&cell) = Vec2::splat(0);
            return;
        }
        let obj = cell.at(obj);
        let diff = *cell - objects.position.expr(&obj).round().cast_i32();
        let angle = objects.angle.expr(&obj);
        let predicted_angle = objects.predicted_angle.expr(&obj);
        let inverted_diff = skew_rotate_quadrant(quadrant_rotate(diff, -quadrant(angle)), -angle);
        let rotated_diff = quadrant_rotate(
            skew_rotate_quadrant(inverted_diff, predicted_angle),
            quadrant(predicted_angle),
        );
        let predicted_pos = objects.predicted_position.expr(&obj).round().cast_i32() + rotated_diff;
        let predicted_cell = cell.at(predicted_pos);

        *physics.delta.var(&cell) = predicted_pos - *cell;

        if physics.lock.atomic(&predicted_cell).fetch_add(1) == 0 {
            *physics.locked_revmap.var(&cell) = *cell - predicted_pos;
            *physics.predicted_object.var(&cell) = *obj;
        } else {
            let index = collisions.next.atomic().fetch_add(1);
            *collisions.data.var(&cell.at(index)) = Collision::from_comps_expr(CollisionComps {
                a_position: *cell,
                predicted_collision: *predicted_cell,
            });
        }
    })
}

#[kernel]
fn collide_kernel(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    collisions: Res<CollisionFields>,
    objects: Res<ObjectFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &collisions.domain, &|el| {
        let collision = collisions.data.expr(&el);
        let a = el.at(collision.a_position);
        let a_obj = el.at(physics.object.expr(&a));
        let pos = collision.predicted_collision;
        let b = el.at(physics.locked_revmap.expr(&el.at(pos)) + pos);
        let b_obj = el.at(physics.object.expr(&b));
        // From a to b.
        let normal = rotate(
            physics.rejection.expr(&a).cast_f32(),
            objects.predicted_angle.expr(&a_obj) - objects.angle.expr(&a_obj),
        ) - rotate(
            physics.rejection.expr(&b).cast_f32(),
            objects.predicted_angle.expr(&b_obj) - objects.angle.expr(&b_obj),
        );
        let force = normal * 0.5;
        let a_impulse = *objects.impulse.atomic(&a_obj);
        a_impulse.x.fetch_sub(force.x);
        a_impulse.y.fetch_sub(force.y);
        let b_impulse = *objects.impulse.atomic(&b_obj);
        b_impulse.x.fetch_add(force.x);
        b_impulse.y.fetch_add(force.y);
    })
}

#[kernel]
fn apply_impulses_kernel(device: Res<Device>, objects: Res<ObjectFields>) -> Kernel<fn()> {
    Kernel::build(&device, &objects.domain, &|obj| {
        *objects.velocity.var(&obj) +=
            objects.impulse.expr(&obj) / objects.mass.expr(&obj).cast_f32();
        *objects.angvel.var(&obj) +=
            objects.angular_impulse.expr(&obj) / objects.moment.expr(&obj).cast_f32();
    })
}

#[kernel]
fn compute_rejection_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let obj = physics.object.expr(&cell);
        if obj == NULL_OBJECT {
            *physics.rejection.var(&cell) = Vec2::splat(0);
            return;
        }
        let best_dist = i32::MAX.var();
        let best_pos = Vec2::splat_expr(0_i32).var();
        for dir in GridDirection::iter_all() {
            let neighbor = world.in_dir(&cell, dir);
            let neighbor_pos = if physics.object.expr(&neighbor) == obj {
                physics.prev_rejection.expr(&neighbor)
            } else {
                Vec2::splat_expr(0)
            } + dir.as_vec();
            if physics.object.expr(&cell.at(neighbor_pos + *cell)) != obj {
                let dist = neighbor_pos.x * neighbor_pos.x + neighbor_pos.y * neighbor_pos.y;
                if dist <= best_dist {
                    *best_dist = dist;
                    *best_pos = neighbor_pos;
                    // TODO: If equal, cancel out. Have to also prevent feedback from farther away things.
                }
            }
        }
        *physics.rejection.var(&cell) = best_pos;
    })
}

#[kernel]
fn copy_rejection_kernel(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    world: Res<World>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *physics
            .prev_rejection
            .var(&cell.at(*cell + physics.delta.expr(&cell))) = physics.rejection.expr(&cell);
    })
}

// #[kernel]
// fn compute_mass(
//     device: Res<Device>,
//     objects: Res<ObjectFields>,
//     physics: Res<PhysicsFields>,
//     world: Res<World>,
// ) -> Kernel<fn()> {
//     Kernel::build(&device, &**world, &|cell| {
//         let obj = cell.at(physics.object.expr(&cell));
//         objects.mass.atomic(&obj).fetch_add(1);
//     })
// }
//
// #[kernel]
// fn

fn init_physics(
    init_data: Res<InitData>,
    objects: Res<ObjectFields>,
    physics: Res<PhysicsFields>,
) -> impl AsNodes {
    let cells = (0..256 * 256)
        .map(|i| {
            let (x, y) = deinterleave_morton(i);
            init_data.cells[x as usize][y as usize]
        })
        .collect::<Vec<_>>();
    let mut object_masses = vec![0_u32; NUM_OBJECTS];
    let mut object_centers = vec![Vector2::repeat(0_u32); NUM_OBJECTS];
    for x in 0..256 {
        for y in 0..256 {
            let obj = init_data.cells[x][y];
            if obj == NULL_OBJECT {
                continue;
            }
            object_masses[obj as usize] += 1;
            object_centers[obj as usize] += Vector2::new(x as u32, y as u32);
        }
    }

    let object_centers = object_centers
        .into_iter()
        .enumerate()
        .map(|(i, sum)| sum.cast::<i32>() / (object_masses[i] as i32).max(1))
        .collect::<Vec<_>>();
    let object_positions = object_centers
        .iter()
        .map(|c| Vec2::from(c.cast::<f32>()))
        .collect::<Vec<_>>();

    let object_velocities = init_data
        .object_velocities
        .iter()
        .map(|v| Vec2::from(*v))
        .chain(repeat(Vec2::splat(0.0)))
        .take(NUM_OBJECTS)
        .collect::<Vec<_>>();
    let mut object_moments = vec![0_u32; NUM_OBJECTS];
    for x in 0..256 {
        for y in 0..256 {
            let obj = init_data.cells[x][y];
            if obj == NULL_OBJECT {
                continue;
            }
            let delta = Vector2::new(x as i32, y as i32) - object_centers[obj as usize];
            let mass = 1;
            let moment = mass * (delta.x * delta.x + delta.y * delta.y) as u32;
            object_moments[obj as usize] += moment;
        }
    }
    let mut object_angvels = init_data.object_angvels.clone();
    object_angvels.resize(NUM_OBJECTS, 0.0);
    (
        objects.buffers.mass.copy_from_vec(object_masses),
        objects.buffers.moment.copy_from_vec(object_moments),
        objects.buffers.position.copy_from_vec(object_positions),
        objects.buffers.angle.copy_from_vec(vec![0.0; NUM_OBJECTS]),
        objects.buffers.velocity.copy_from_vec(object_velocities),
        objects.buffers.angvel.copy_from_vec(object_angvels),
        physics.object_buffer.copy_from_vec(cells),
    )
}

fn update_physics(collisions: Res<CollisionFields>, physics: Res<PhysicsFields>) -> impl AsNodes {
    (
        // TODO: Move to end, currently here for debugging.
        physics
            .lock_buffer
            .copy_from_vec(vec![0; physics.lock_buffer.len()]),
        predict_kernel.dispatch(),
        move_kernel.dispatch(),
        collisions.next.read_to(&collisions.domain.len),
        // collide_kernel.dispatch(),
        // apply_impulses_kernel.dispatch(),
        finalize_objects_kernel.dispatch(),
        finalize_move_kernel.dispatch(),
        // Reset lock, buffers.
        collisions.next.write_host(0),
        copy_rejection_kernel.dispatch(),
        compute_rejection_kernel.dispatch(),
    )
        .chain()
}

pub struct PhysicsPlugin;
impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_objects, setup_physics))
            .add_systems(
                InitKernel,
                (
                    init_clear_objects_kernel,
                    init_predict_kernel,
                    init_finalize_objects_kernel,
                    init_finalize_move_kernel,
                    init_move_kernel,
                    init_collide_kernel,
                    init_apply_impulses_kernel,
                    init_compute_rejection_kernel,
                    init_copy_rejection_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(init_physics))
            .add_systems(WorldUpdate, add_update(update_physics));
    }
}
