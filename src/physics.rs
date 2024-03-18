use bevy::utils::HashMap;
use id_newtype::UniqueId;
use morton::interleave_morton;
use rapier2d::prelude::*;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, UniqueId)]
pub struct ObjectId(u32);

#[derive(Resource)]
struct ObjectFields {
    // TODO: Change for resizing.
    pub domain: StaticDomain<1>,
    pub position: VField<Vec2<f32>, u32>,
    pub velocity: VField<Vec2<f32>, u32>,
    pub angle: VField<f32, u32>,
    pub angvel: VField<f32, u32>,
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
    pub object: VField<u32, Vec2<i32>>,
    pub velocity: VField<Vec2<f32>, Vec2<i32>>,
    _fields: FieldSet,
    object_buffer: Buffer<u32>,
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
    let object = *fields.create_bind(
        "physics-objects",
        world.map_buffer_morton(object_buffer.view(..)),
    );
    let velocity = fields.create_bind("physics-velocity", world.create_texture(&device));

    let physics = PhysicsFields {
        object,
        velocity,
        _fields: fields,
        object_buffer,
    };

    commands.insert_resource(objects);
    commands.insert_resource(physics);
}

#[derive(Resource, Default)]
struct ObjectFieldStaging(Option<Vec<u32>>);

fn compute_object_staging(
    rb_context: Res<RigidBodyContext>,
    mut staging: ResMut<ObjectFieldStaging>,
) {
    // TODO: Do something else since this is just dumb.
    assert!(staging.0.is_none());
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

    staging.0 = Some(values);
}

fn update_objects(
    physics: Res<PhysicsFields>,
    mut staging: ResMut<ObjectFieldStaging>,
    mut allowed_run: Local<bool>,
) -> impl AsNodes {
    if !*allowed_run {
        *allowed_run = true;
        return ().into_node_configs();
    }
    let staging = staging.0.take().unwrap();

    physics
        .object_buffer
        .copy_from_vec(staging)
        .into_node_configs()
}

fn update_bodies(mut rb_context: ResMut<RigidBodyContext>) {
    rb_context.step();
}

pub struct PhysicsPlugin;
impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RigidBodyContext {
            gravity: Vector2::new(0.0, -20.0),
            ..default()
        })
        .init_resource::<ObjectFieldStaging>()
        .add_systems(Startup, setup_physics)
        .add_systems(
            WorldUpdate,
            add_update(update_objects).in_set(UpdatePhase::CopyBodiesFromHost),
        )
        .add_systems(HostUpdate, (update_bodies, compute_object_staging));
    }
}
