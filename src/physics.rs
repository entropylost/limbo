use bevy::utils::HashMap;
use bevy_sefirot::MirrorGraph;
use id_newtype::UniqueId;
use morton::deinterleave_morton;
use rapier2d::prelude::*;
use sefirot::mapping::buffer::StaticDomain;
use sefirot::mapping::function::CachedFnMapping;
use sefirot_grid::GridDomain;

use crate::prelude::*;
use crate::render::RenderFields;

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct PhysicsGraph(pub MirrorGraph);
impl FromWorld for PhysicsGraph {
    fn from_world(world: &mut World) -> Self {
        PhysicsGraph(MirrorGraph::null(world.resource::<Device>()))
    }
}

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

const NULL_OBJECT: u32 = u32::MAX;

#[derive(Resource)]
pub struct PhysicsFields {
    pub domain: GridDomain,
    pub object: VField<u32, Vec2<i32>>,
    pub velocity: VField<Vec2<f32>, Vec2<i32>>,
    _fields: FieldSet,
    object_buffer: Buffer<u32>,
}

#[derive(Resource)]
struct ImfFields {
    value: AField<u32, Vec2<i32>>,
    next_value: AField<u32, Vec2<i32>>,
    out: VField<Vec2<i32>, Vec2<i32>>,
    valid: VField<bool, Vec2<i32>>,
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

fn setup_physics(mut commands: Commands, device: Res<Device>, rb_context: Res<RigidBodyContext>) {
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
    let physics_domain = GridDomain::new([-64, -64], [256, 256]);
    let mut physics_fields = FieldSet::new();
    let object_buffer = device.create_buffer(256 * 256);
    let object = *physics_fields.create_bind(
        "physics-objects",
        physics_domain.map_buffer_morton(object_buffer.view(..)),
    );
    let velocity = physics_fields.create_bind(
        "physics-velocity",
        physics_domain.map_texture(device.create_tex2d(PixelStorage::Float2, 256, 256, 1)),
    );

    let imf = ImfFields {
        value: physics_fields.create_bind(
            "imf-value",
            physics_domain.map_buffer_morton(device.create_buffer(256 * 256)),
        ),
        next_value: physics_fields.create_bind(
            "imf-value",
            physics_domain.map_buffer_morton(device.create_buffer(256 * 256)),
        ),
        out: physics_fields.create_bind(
            "imf-out",
            physics_domain.map_texture(device.create_tex2d(PixelStorage::Int2, 256, 256, 1)),
        ),
        valid: *physics_fields.create_bind(
            "imf-valid",
            physics_domain.map_buffer_morton(device.create_buffer(256 * 256)),
        ),
    };

    let physics = PhysicsFields {
        domain: physics_domain,
        object,
        velocity,
        _fields: physics_fields,
        object_buffer,
    };

    commands.insert_resource(objects);
    commands.insert_resource(physics);
    commands.insert_resource(imf);
}

fn update_objects(physics: Res<PhysicsFields>, rb_context: Res<RigidBodyContext>) {
    use rayon::prelude::*;
    // TODO: Do something else since this is just dumb.
    let staging = (0_u32..256 * 256)
        .into_par_iter()
        .map(|i| {
            let (x, y) = deinterleave_morton(i);
            let pos = Vector2::new(x, y).cast::<f32>() + Vector2::repeat(0.5 - 64.0);
            let mut object = NULL_OBJECT;
            rb_context.query_pipeline.intersections_with_point(
                &rb_context.bodies,
                &rb_context.colliders,
                &Point::from(pos),
                QueryFilter::default(),
                |handle| {
                    if let Some(handle) = rb_context.colliders[handle].parent() {
                        let obj = rb_context.object_map[&handle];
                        object = object.min(obj.0);
                    }
                    true
                },
            );
            object
        })
        .collect::<Vec<_>>();
    // TODO: Asynchronize.
    physics.object_buffer.copy_from(&staging);
}

fn update_bodies(mut rb_context: ResMut<RigidBodyContext>) {
    rb_context.step();
}
const IMF_CAP: u32 = 2048;

#[kernel]
fn render_kernel(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
    render: Res<RenderFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &render.domain, &|mut el| {
        let mut world_el = physics.domain.index(el.cast_i32(), &el);
        let object = world_el.expr(&physics.object);
        let color = // if object == NULL_OBJECT {
            Vec3::expr(1.0, 0.2, 0.0) * world_el.expr(&imf.value).as_f32() / IMF_CAP as f32;
        // } else {
        //     Vec3::splat_expr(1.0)
        // };
        *el.var(&render.color) = color;
    })
}

#[kernel]
fn update_valid(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &physics.domain, &|mut el| {
        *el.var(&imf.valid) = physics
            .domain
            .index(el.expr(&imf.out), &el)
            .expr(&imf.value)
            < IMF_CAP;
    })
}

#[kernel]
fn init_imf_out(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &physics.domain, &|mut el| {
        *el.var(&imf.out) = *el;
    })
}

#[kernel]
fn propegate_imf_out(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &physics.domain, &|mut el| {
        let best_dist = i32::MAX.var();
        let best_out = (*el).var();
        let pos = *el;
        physics.domain.on_adjacent(&el, |mut el| {
            if el.expr(&imf.valid) {
                let out = el.expr(&imf.out);
                let delta = out - pos;
                let dist = delta.x * delta.x + delta.y * delta.y;
                if dist < best_dist {
                    *best_dist = dist;
                    *best_out = out;
                }
            }
        });
        let out = el.expr(&imf.out);
        if el.expr(&imf.valid) {
            let delta = out - pos;
            let dist = delta.x * delta.x + delta.y * delta.y;
            if dist < best_dist {
                *best_dist = dist;
                *best_out = out;
            }
        }
        if el.expr(&imf.value) < IMF_CAP / 2 {
            *best_out = pos;
            *best_dist = 0;
        }
        // TODO: Also check the current out to see if it's also good?
        if best_dist < i32::MAX {
            *el.var(&imf.out) = best_out;
        }
    })
}

#[kernel]
fn imf_kernel(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &physics.domain, &|mut el| {
        let object = el.expr(&physics.object);
        let value = el.expr(&imf.value);
        let next_value = el.atomic(&imf.next_value);
        if object == 1 {
            // Player
            next_value.fetch_add(IMF_CAP / 16);
        };
        if value > IMF_CAP && el.expr(&imf.valid) {
            let diff = value - IMF_CAP;
            next_value.fetch_sub(diff);
            let mut out = physics.domain.index(el.expr(&imf.out), &el);
            out.atomic(&imf.next_value).fetch_add(diff);
        }
        if value >= 1 {
            next_value.fetch_sub(1);
        }
    })
}

#[kernel]
fn copy_next_imf_kernel(
    device: Res<Device>,
    physics: Res<PhysicsFields>,
    imf: Res<ImfFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &physics.domain, &|mut el| {
        *el.var(&imf.value) = el.expr(&imf.next_value);
    })
}

fn run_physics(mut graph: ResMut<PhysicsGraph>, mut ran_first: Local<bool>) {
    if !*ran_first {
        graph.add(
            (
                init_imf_out.dispatch(),
                update_valid.dispatch(),
                render_kernel.dispatch(),
            )
                .chain(),
        );
        *ran_first = true;
    } else {
        graph.add(
            (
                propegate_imf_out.dispatch(),
                update_valid.dispatch(),
                imf_kernel.dispatch(),
                copy_next_imf_kernel.dispatch(),
                render_kernel.dispatch(),
            )
                .chain(),
        );
    }
    graph.execute_clear();
}

pub struct PhysicsPlugin;
impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RigidBodyContext {
            // gravity: Vector2::new(0.0, -20.0),
            ..default()
        })
        .init_resource::<PhysicsGraph>()
        .add_systems(
            InitKernel,
            (
                init_render_kernel,
                init_init_imf_out,
                init_update_valid,
                init_propegate_imf_out,
                init_imf_kernel,
                init_copy_next_imf_kernel,
            ),
        )
        .add_systems(Startup, setup_physics)
        .add_systems(Update, (update_bodies, update_objects, run_physics).chain());
    }
}
