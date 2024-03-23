use std::f32::consts::TAU;

use bevy::utils::HashMap;
use id_newtype::UniqueId;
use morton::interleave_morton;
use rapier2d::prelude::*;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;

const NUM_OBJECTS: u32 = 16;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, UniqueId)]
#[repr(transparent)]
pub struct ObjectHost(u32);

pub type Object = Expr<u32>;

#[derive(Resource)]
pub struct ObjectFields {
    // TODO: Change for resizing.
    pub domain: StaticDomain<1>,
    // Also change these to use ObjectId instead.
    pub position: VField<Vec2<i32>, Object>,
    pub last_position: VField<Vec2<i32>, Object>,
    pub velocity: VField<Vec2<f32>, Object>,
    pub angle: VField<f32, Object>,
    pub last_angle: VField<f32, Object>,
    pub angvel: VField<f32, Object>,
    _fields: FieldSet,
    buffers: ObjectBuffers,
}

struct ObjectBuffers {
    position: Buffer<Vec2<i32>>,
    last_position: Buffer<Vec2<i32>>,
    velocity: Buffer<Vec2<f32>>,
    angle: Buffer<f32>,
    last_angle: Buffer<f32>,
    angvel: Buffer<f32>,
}

pub const NULL_OBJECT: u32 = u32::MAX;

#[derive(Resource)]
pub struct PhysicsFields {
    pub object: VField<u32, Cell>,
    object_staging: VField<u32, Cell>,
    pub delta: VField<Vec2<i32>, Cell>,
    pub velocity: VField<Vec2<f32>, Cell>,
    _fields: FieldSet,
    object_buffer: Buffer<u32>,
    object_staging_buffer: Buffer<u32>,
}

#[derive(Default, Resource)]
pub struct RigidBodyContext {
    pub gravity: Vector2<f32>,
    pub bodies: RigidBodySet,
    pub object_map: HashMap<RigidBodyHandle, ObjectHost>,
    pub colliders: ColliderSet,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub islands: IslandManager,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    pub physics_hooks: (),
    pub event_handler: (),
}

impl RigidBodyContext {
    fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &self.physics_hooks,
            &self.event_handler,
        );
    }
    pub fn insert2(&mut self, body: RigidBody, collider: Collider) -> RigidBodyHandle {
        let body = self.bodies.insert(body);
        let _collider = self
            .colliders
            .insert_with_parent(collider, body, &mut self.bodies);
        self.object_map.insert(body, ObjectHost::unique());
        body
    }
}

fn setup_physics(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let obj_domain = StaticDomain::<1>::new(NUM_OBJECTS);
    let mut obj_fields = FieldSet::new();
    let position_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let last_position_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let velocity_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let angle_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let last_angle_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let angvel_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let position = obj_fields.create_bind(
        "object-position",
        obj_domain.map_buffer(position_buffer.view(..)),
    );
    let last_position = obj_fields.create_bind(
        "object-last-position",
        obj_domain.map_buffer(last_position_buffer.view(..)),
    );
    let velocity = obj_fields.create_bind(
        "object-velocity",
        obj_domain.map_buffer(velocity_buffer.view(..)),
    );
    let angle =
        obj_fields.create_bind("object-angle", obj_domain.map_buffer(angle_buffer.view(..)));
    let last_angle = obj_fields.create_bind(
        "object-last-angle",
        obj_domain.map_buffer(last_angle_buffer.view(..)),
    );
    let angvel = obj_fields.create_bind(
        "object-angvel",
        obj_domain.map_buffer(angvel_buffer.view(..)),
    );
    let object_buffers = ObjectBuffers {
        position: position_buffer,
        last_position: last_position_buffer,
        last_angle: last_angle_buffer,
        velocity: velocity_buffer,
        angle: angle_buffer,
        angvel: angvel_buffer,
    };
    let objects = ObjectFields {
        domain: obj_domain,
        position,
        last_position,
        last_angle,
        velocity,
        angle,
        angvel,
        _fields: obj_fields,
        buffers: object_buffers,
    };
    let mut fields = FieldSet::new();
    let object_buffer = device.create_buffer((world.width() * world.height()) as usize);
    let object_staging_buffer = device.create_buffer((world.width() * world.height()) as usize);
    let object = *fields.create_bind("physics-object", world.map_buffer(object_buffer.view(..)));
    let object_staging = *fields.create_bind(
        "physics-object-staging",
        world.map_buffer(object_staging_buffer.view(..)),
    );
    let delta = fields.create_bind("physics-delta", world.create_texture(&device));
    let velocity = fields.create_bind("physics-velocity", world.create_texture(&device));

    let physics = PhysicsFields {
        object,
        object_staging,
        delta,
        velocity,
        _fields: fields,
        object_buffer,
        object_staging_buffer,
    };

    commands.insert_resource(objects);
    commands.insert_resource(physics);
}

#[derive(Resource, Default)]
struct ExtractResource(Option<ExtractData>);

struct ExtractData {
    physics_objects: Vec<u32>,
    object_position: Vec<Vec2<i32>>,
    object_velocity: Vec<Vec2<f32>>,
    object_angle: Vec<f32>,
    object_angvel: Vec<f32>,
}

fn compute_object_staging(
    rb_context: Res<RigidBodyContext>,
    mut staging_res: ResMut<ExtractResource>,
) {
    // TODO: Do something else since this is just dumb.
    assert!(staging_res.0.is_none());
    let mut objects = vec![NULL_OBJECT; 256 * 256];

    // Shouldn't actually use this. Use previous frame with object separations.
    for (_handle, collider) in rb_context.colliders.iter() {
        let object = rb_context.object_map[&collider.parent().unwrap()].0;
        let aabb = collider.compute_aabb();
        let min = aabb.mins.map(|x| x.floor() as i32);
        let max = aabb.maxs.map(|x| x.ceil() as i32);
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                let pos = Vector2::new(x, y).cast::<f32>() + Vector2::repeat(0.5);
                if collider
                    .shape()
                    .contains_point(collider.position(), &Point::from(pos))
                {
                    let data_pos = Vector2::new(x, y) + Vector2::repeat(64);
                    let data_pos = data_pos.map(|x| x.rem_euclid(256));
                    let i = interleave_morton(data_pos.x as u16, data_pos.y as u16);
                    objects[i as usize] = objects[i as usize].min(object);
                }
            }
        }
    }
    let mut staging = ExtractData {
        physics_objects: objects,
        object_position: vec![Vec2::splat(0); 16],
        object_velocity: vec![Vec2::splat(0.0); 16],
        object_angle: vec![0.0; 16],
        object_angvel: vec![0.0; 16],
    };
    for (handle, body) in rb_context.bodies.iter() {
        let object = rb_context.object_map[&handle].0;
        let i = object as usize;
        staging.object_position[i] = Vec2::from(body.translation().map(|x| x.round() as i32));
        staging.object_velocity[i] = Vec2::from(*body.linvel());
        staging.object_angle[i] = body.rotation().angle();
        staging.object_angvel[i] = body.angvel();
    }
    staging_res.0 = Some(staging);
}

fn extract_to_device(
    physics: Res<PhysicsFields>,
    objects: Res<ObjectFields>,
    mut staging_res: ResMut<ExtractResource>,
    mut not_first: Local<bool>,
) -> impl AsNodes {
    let staging = staging_res.0.take().unwrap();

    let extract = (
        (
            objects
                .buffers
                .position
                .copy_to_buffer_async(&objects.buffers.last_position),
            objects
                .buffers
                .position
                .copy_from_vec(staging.object_position.clone()),
        )
            .chain(),
        objects
            .buffers
            .velocity
            .copy_from_vec(staging.object_velocity),
        (
            objects
                .buffers
                .angle
                .copy_to_buffer_async(&objects.buffers.last_angle),
            objects.buffers.angle.copy_from_vec(staging.object_angle),
        )
            .chain(),
        objects.buffers.angvel.copy_from_vec(staging.object_angvel),
    );
    if *not_first {
        (
            physics
                .object_buffer
                .copy_to_buffer_async(&physics.object_staging_buffer),
            extract,
        )
            .into_node_configs()
    } else {
        *not_first = true;
        (
            physics
                .object_staging_buffer
                .copy_from_vec(staging.physics_objects),
            extract,
            objects
                .buffers
                .last_position
                .copy_from_vec(staging.object_position),
        )
            .chain()
            .into_node_configs()
    }
}

// fn extract_to_host() -> impl AsNodes {
//     todo!();
// }

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
fn quadrant(angle: Expr<f32>) -> Expr<i32> {
    (angle * 4.0 / TAU).round().cast_i32().rem_euclid(4)
}

#[kernel(run)]
fn clear_objects_kernel(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    world: Res<World>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *physics.object.var(&cell) = NULL_OBJECT;
    })
}

#[kernel(run)]
fn update_objects_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
    objects: Res<ObjectFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        // TODO: What to do about collisions?
        let obj = physics.object_staging.expr(&cell);
        if obj == NULL_OBJECT {
            return;
        }
        let obj = cell.at(obj);
        let diff = *cell - objects.last_position.expr(&obj);
        let last_angle = objects.last_angle.expr(&obj);
        let angle = objects.angle.expr(&obj);
        let inverted_diff =
            skew_rotate_quadrant(quadrant_rotate(diff, -quadrant(last_angle)), -last_angle);
        let rotated_diff =
            quadrant_rotate(skew_rotate_quadrant(inverted_diff, angle), quadrant(angle));
        let new_pos = objects.position.expr(&obj) + rotated_diff;
        let new_cell = cell.at(new_pos);
        *physics.object.var(&new_cell) = *obj;
        *physics.delta.var(&cell) = new_pos - *cell;
        *physics.velocity.var(&new_cell) = objects.velocity.expr(&obj)
            + objects.angvel.expr(&obj) * Vec2::expr(-rotated_diff.y, rotated_diff.x).cast_f32();
    })
}

fn update_bodies(mut rb_context: ResMut<RigidBodyContext>) {
    rb_context.step();
}

pub struct PhysicsPlugin;
impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RigidBodyContext {
            gravity: Vector2::new(0.0, 0.0),
            integration_parameters: IntegrationParameters {
                dt: 1.0,
                min_ccd_dt: 1.0 / 100.0,
                ..default()
            },
            ..default()
        })
        .init_resource::<ExtractResource>()
        .add_systems(Startup, setup_physics)
        .add_systems(
            InitKernel,
            (init_update_objects_kernel, init_clear_objects_kernel),
        )
        .add_systems(PostStartup, compute_object_staging)
        .add_systems(
            WorldUpdate,
            (
                add_update(extract_to_device),
                add_update(clear_objects),
                add_update(update_objects),
            )
                .chain()
                .in_set(UpdatePhase::CopyBodiesFromHost),
        )
        .add_systems(HostUpdate, (update_bodies, compute_object_staging).chain());
    }
}
