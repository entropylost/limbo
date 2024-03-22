use std::f32::consts::TAU;

use bevy::utils::HashMap;
use id_newtype::UniqueId;
use morton::interleave_morton;
use rapier2d::prelude::*;
use sefirot::mapping::buffer::StaticDomain;
use sefirot::utils::Singleton;

use crate::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, UniqueId)]
#[repr(transparent)]
pub struct ObjectId(u32);

#[derive(Resource)]
struct ObjectFields {
    // TODO: Change for resizing.
    pub domain: StaticDomain<1>,
    // Also change these to use ObjectId instead.
    pub position: VEField<Vec2<f32>, u32>,
    pub velocity: VEField<Vec2<f32>, u32>,
    pub angle: VEField<f32, u32>,
    pub angvel: VEField<f32, u32>,
    _fields: FieldSet,
    buffers: ObjectBuffers,
}

struct ObjectBuffers {
    position: Buffer<Vec2<f32>>,
    velocity: Buffer<Vec2<f32>>,
    angle: Buffer<f32>,
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
}

#[derive(Resource)]
pub struct SolidFields {
    solid_domain: StaticDomain<1>,
    solid_index: VField<u32, Cell>,
    solid_counter: Singleton<u32>,
    parent: AEField<u32, u32>,
}

#[derive(Default, Resource)]
pub struct RigidBodyContext {
    pub gravity: Vector2<f32>,
    pub bodies: RigidBodySet,
    pub object_map: HashMap<RigidBodyHandle, ObjectId>,
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
        self.object_map.insert(body, ObjectId::unique());
        body
    }
}

fn setup_physics(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    const NUM_OBJECTS: u32 = 16;
    let obj_domain = StaticDomain::<1>::new(NUM_OBJECTS);
    let mut obj_fields = FieldSet::new();
    let position_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let velocity_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let angle_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let angvel_buffer = device.create_buffer(NUM_OBJECTS as usize);
    let position = obj_fields.create_bind(
        "object-position",
        obj_domain.map_buffer(position_buffer.view(..)),
    );
    let velocity = obj_fields.create_bind(
        "object-velocity",
        obj_domain.map_buffer(velocity_buffer.view(..)),
    );
    let angle =
        obj_fields.create_bind("object-angle", obj_domain.map_buffer(angle_buffer.view(..)));
    let angvel = obj_fields.create_bind(
        "object-angvel",
        obj_domain.map_buffer(angvel_buffer.view(..)),
    );
    let object_buffers = ObjectBuffers {
        position: position_buffer,
        velocity: velocity_buffer,
        angle: angle_buffer,
        angvel: angvel_buffer,
    };
    let objects = ObjectFields {
        domain: obj_domain,
        position,
        velocity,
        angle,
        angvel,
        _fields: obj_fields,
        buffers: object_buffers,
    };
    let mut fields = FieldSet::new();
    let object_buffer = device.create_buffer(256 * 256);
    let object = *fields.create_bind("physics-object", world.map_buffer(object_buffer.view(..)));
    let object_staging =
        *fields.create_bind("physics-object-staging", world.create_buffer(&device));
    let delta = fields.create_bind("physics-delta", world.create_texture(&device));
    let velocity = fields.create_bind("physics-velocity", world.create_texture(&device));

    let solid_domain = StaticDomain::<1>::new(6000); // Random number really.
    let solid_index = *fields.create_bind("solid-index", world.create_buffer(&device));
    let solid_counter = Singleton::new(&device);
    let parent = fields.create_bind("solid-parent", solid_domain.create_buffer(&device));

    let physics = PhysicsFields {
        object,
        object_staging,
        delta,
        velocity,
        _fields: fields,
        object_buffer,
    };

    let solids = SolidFields {
        solid_domain,
        solid_index,
        solid_counter,
        parent,
    };

    commands.insert_resource(objects);
    commands.insert_resource(physics);
    commands.insert_resource(solids);
}

#[derive(Resource, Default)]
struct ObjectFieldStaging {
    physics_objects: Option<Vec<u32>>,
    object_velocity: Option<Vec<Vec2<f32>>>,
}

fn compute_object_staging(
    rb_context: Res<RigidBodyContext>,
    mut staging: ResMut<ObjectFieldStaging>,
) {
    // TODO: Do something else since this is just dumb.
    assert!(staging.physics_objects.is_none());
    let mut values = vec![NULL_OBJECT; 256 * 256];

    for (_handle, collider) in rb_context.colliders.iter() {
        let object = rb_context.object_map[&collider.parent().unwrap()].0;
        let aabb = collider.compute_aabb();
        let min = aabb.mins.map(|x| x.round() as i32);
        let max = aabb.maxs.map(|x| x.round() as i32);
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
                    values[i as usize] = values[i as usize].min(object);
                }
            }
        }
    }

    staging.physics_objects = Some(values);

    assert!(staging.object_velocity.is_none());
    let mut velocities = rb_context
        .bodies
        .iter()
        .map(|(_, body)| Vec2::from(*body.linvel()))
        .collect::<Vec<_>>();
    velocities.resize(16, Vec2::splat(0.0));
    staging.object_velocity = Some(velocities);
}

fn update_objects(
    physics: Res<PhysicsFields>,
    objects: Res<ObjectFields>,
    mut staging: ResMut<ObjectFieldStaging>,
) -> impl AsNodes {
    if staging.physics_objects.is_none() {
        return ().into_node_configs();
    }
    let staging_objects = staging.physics_objects.take().unwrap();
    let staging_velocity = staging.object_velocity.take().unwrap();

    (
        physics.object_buffer.copy_from_vec(staging_objects),
        objects.buffers.velocity.copy_from_vec(staging_velocity),
    )
        .into_node_configs()
}

#[kernel]
fn skew_x_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn(f32)> {
    Kernel::build(&device, &**world, &|cell, skew| {
        *physics.next_object.var(&cell) = physics.object.expr(&cell.at(Vec2::expr(
            cell.x - (cell.y.cast_f32() * skew).round().cast_i32(),
            cell.y,
        )));
    })
}
#[kernel]
fn skew_y_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn(f32)> {
    Kernel::build(&device, &**world, &|cell, skew| {
        *physics.next_object.var(&cell) = physics.object.expr(&cell.at(Vec2::expr(
            cell.x,
            cell.y - (cell.x.cast_f32() * skew).round().cast_i32(),
        )));
    })
}

#[kernel]
fn copyback_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *physics.object.var(&cell) = physics.next_object.expr(&cell);
    })
}

fn world_rotate(mut angle: Local<f32>) -> impl AsNodes {
    // let last_angle = *angle;
    *angle += 0.1 * TAU / 360.0;
    println!("Angle: {:?}", angle);
    (
        // skew_x_kernel.dispatch(&(*angle / 2.0).tan()),
        // copyback_kernel.dispatch(),
        // skew_y_kernel.dispatch(&-angle.sin()),
        // copyback_kernel.dispatch(),
        // skew_x_kernel.dispatch(&(*angle / 2.0).tan()),
        // copyback_kernel.dispatch(),
        skew_x_kernel.dispatch(&-(*angle / 2.0).tan()),
        copyback_kernel.dispatch(),
        skew_y_kernel.dispatch(&angle.sin()),
        copyback_kernel.dispatch(),
        skew_x_kernel.dispatch(&-(*angle / 2.0).tan()),
        copyback_kernel.dispatch(),
    )
        .chain()
}

#[kernel(run)]
fn derive_velocity_kernel(
    device: Res<Device>,
    world: Res<World>,
    physics: Res<PhysicsFields>,
    objects: Res<ObjectFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|el| {
        let obj = physics.object.expr(&el);
        if obj == NULL_OBJECT {
            return;
        }
        let vel = objects.velocity.expr(&el.at(obj));
        *physics.velocity.var(&el) = vel;
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
        .init_resource::<ObjectFieldStaging>()
        .add_systems(Startup, setup_physics)
        .add_systems(
            InitKernel,
            (
                init_copyback_kernel,
                init_skew_x_kernel,
                init_skew_y_kernel,
                init_derive_velocity_kernel,
            ),
        )
        .add_systems(PostStartup, compute_object_staging)
        .add_systems(
            WorldUpdate,
            (add_update(update_objects), add_update(world_rotate)).chain(),
        )
        .add_systems(HostUpdate, (update_bodies, compute_object_staging).chain());
    }
}
