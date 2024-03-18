use std::f32::consts::{PI, TAU};

use luisa::lang::functions::sync_block;
use luisa::lang::types::shared::Shared;
use sefirot::mapping::buffer::StaticDomain;

use super::prelude::*;
use crate::physics::{PhysicsFields, NULL_OBJECT};
pub use crate::prelude::*;
use crate::utils::rand_f32;

#[derive(Resource)]
pub struct LightFields {
    pub light_domain: StaticDomain<1>,
    pub domain: StaticDomain<2>,
    trace_domain: StaticDomain<2>,
    _entire_domain: StaticDomain<3>,
    pub wall: VField<u32, Vec2<u32>>,
    pub radiance: VField<Vec3<f32>, Vec3<u32>>,
    pub sunlight: VField<Vec3<f32>, u32>,
    _fields: FieldSet,
}

fn setup_fields(mut commands: Commands, device: Res<Device>, constants: Res<LightConstants>) {
    let skylight = constants
        .skylight
        .iter()
        .map(|v| Vec3::from(*v))
        .collect::<Vec<_>>();

    let light_domain = StaticDomain::<1>::new(constants.directions);
    let domain = StaticDomain::<2>::new(constants.trace_size, constants.trace_size);
    let trace_domain = StaticDomain::<2>::new(constants.trace_size, constants.directions);
    let entire_domain = StaticDomain::<3>::new(
        constants.trace_size,
        constants.trace_size,
        constants.directions,
    );
    let mut fields = FieldSet::new();
    let wall = fields.create_bind("light-wall", domain.create_tex2d(&device));
    let radiance = fields.create_bind("light-radiance", entire_domain.create_tex3d(&device));
    let sunlight = fields.create_bind(
        "sunlight",
        light_domain.map_buffer(device.create_buffer_from_slice(&skylight)),
    );
    commands.insert_resource(LightFields {
        light_domain,
        domain,
        trace_domain,
        _entire_domain: entire_domain,
        wall,
        radiance,
        sunlight,
        _fields: fields,
    });
}

#[kernel]
fn wall_kernel(
    device: Res<Device>,
    world: Res<World>,
    light: Res<LightFields>,
    physics: Res<PhysicsFields>,
) -> Kernel<fn(Vec2<i32>)> {
    Kernel::build(&device, &light.domain, &|el, offset| {
        let world_el = el.at(el.cast_i32() + offset);
        if world.contains(&world_el) {
            let wall = physics.object.expr(&world_el) != NULL_OBJECT;
            *light.wall.var(&el) = wall.cast::<u32>();
        }
    })
}

#[kernel]
fn trace_kernel(
    device: Res<Device>,
    light: Res<LightFields>,
    constants: Res<LightConstants>,
) -> Kernel<fn(u32)> {
    let trace_size = constants.trace_size;
    let blur = constants.blur;
    let directions = constants.directions;
    let trace_length = constants.trace_size;
    let grid_size = constants.trace_size;
    Kernel::build(&device, &light.trace_domain, &|el, t| {
        set_block_size([trace_size, 1, 1]);
        let dir = el.y;
        let index = el.x;

        let angle = (dir.cast_f32() * TAU) / directions as f32 + 0.0001;
        let quadrant = (dir / (directions / 4)) % 4;

        let radiance = light.sunlight.expr(&el.at(dir)).var();

        let ray_dir = Vec2::expr(angle.cos(), angle.sin());
        let delta_dist = 1.0 / ray_dir.abs();
        let step = ray_dir.signum().cast_i32();

        let correction = ray_dir.x.abs() + ray_dir.y.abs();

        let trace_length = correction * correction * trace_length as f32;

        let ray_pos = Vec2::<f32>::splat(grid_size as f32 / 2.0)
            - (trace_length / 2.0) * Vec2::expr(angle.cos(), angle.sin()) / correction
            - (trace_size as f32 / 2.0) * Vec2::expr(-angle.sin(), angle.cos()) * correction
            + Vec2::expr(
                rand_f32(Vec2::expr(dir, t), 0.expr(), 0),
                rand_f32(Vec2::expr(dir, t), 1.expr(), 0),
            )
            + index.cast_f32() * Vec2::expr(-step.y.as_f32(), step.x.as_f32())
            + index.cast_f32()
                * 2.0_f32.sqrt()
                * (quadrant.as_f32() * PI / 2.0 + PI / 4.0 - angle).sin()
                * ray_dir;
        let pos = ray_pos.floor().cast_i32().var();

        let side_dist =
            (ray_dir.signum() * (pos.cast_f32() - ray_pos) + ray_dir.signum() * 0.5 + 0.5)
                * delta_dist;
        let side_dist = side_dist.var();

        // Remove to make the light look manhattan.
        let blur = blur / correction;

        let shared = Shared::<Vec3<f32>>::new(trace_size as usize + 2);

        if dispatch_id().x == 0 {
            shared.write(0, radiance);
            shared.write(trace_size + 1, radiance);
        }

        for _i in 0.expr()..trace_length.cast_u32() {
            let si = index + 1;
            shared.write(si, radiance);
            sync_block();
            *radiance =
                (1.0 - 2.0 * blur) * radiance + blur * (shared.read(si - 1) + shared.read(si + 1));

            let mask = side_dist <= side_dist.yx();
            *side_dist += mask.select(delta_dist, Vec2::splat_expr(0.0));
            *pos += mask.select(step, Vec2::splat_expr(0));

            if pos.x < 0 || pos.y < 0 || pos.y >= grid_size as i32 {
                continue;
            }

            let pos = pos.cast_u32();

            let wall = light.wall.expr(&el.at(pos)) != 0;
            if wall {
                *radiance = Vec3::splat(0.0); // wall / directions as f32;
            }

            *light.radiance.var(&el.at(pos.extend(dir))) = radiance;
        }
    })
}

#[kernel]
fn accumulate_kernel(
    device: Res<Device>,
    world: Res<World>,
    light: Res<LightFields>,
    constants: Res<LightConstants>,
    render: Res<RenderFields>,
) -> Kernel<fn(Vec2<i32>)> {
    Kernel::build(&device, &light.domain, &|el, offset| {
        let radiance = Vec3::splat(0.0_f32).var();
        for dir in 0..constants.directions {
            *radiance += light.radiance.expr(&el.at(el.extend(dir)));
        }
        let world_el = el.at(el.cast_i32() + offset);
        if world.contains(&world_el) {
            *render.color.var(&world_el) = radiance;
        }
    })
}

fn color(parameters: Res<LightParameters>, mut time: Local<u32>) -> impl AsNodes {
    *time = time.wrapping_add(1);
    let offset = Vec2::from(parameters.offset);
    (
        wall_kernel.dispatch(&offset),
        trace_kernel.dispatch(&*time),
        accumulate_kernel.dispatch(&offset),
    )
        .chain()
}

#[derive(Resource, Clone)]
pub struct LightConstants {
    trace_size: u32,
    directions: u32,
    blur: f32,
    skylight: Vec<Vector3<f32>>,
}
impl Default for LightConstants {
    fn default() -> Self {
        let directions = 64;
        Self {
            trace_size: 256,
            directions,
            blur: 0.3,
            skylight: (0..directions)
                .map(|dir| {
                    let angle = (dir as f32 * TAU) / directions as f32;
                    let norm = (-angle.sin()).max(0.0) * (-angle.sin()).max(0.0);
                    let sun: f32 = if dir == 53 { 1.0 } else { 0.0 };
                    Vector3::new(0.3, 0.7, 1.0) * norm * 0.3 / directions as f32
                        + sun * Vector3::new(1.0, 1.0, 0.8) * 0.1
                })
                .collect::<Vec<_>>(),
        }
    }
}

#[derive(Resource, Copy, Clone)]
pub struct LightParameters {
    pub offset: Vector2<i32>,
}
impl Default for LightParameters {
    fn default() -> Self {
        Self {
            offset: Vector2::new(0, 0),
        }
    }
}
impl LightParameters {
    pub fn set_center(&mut self, constants: &LightConstants, center: Vector2<i32>) {
        self.offset = center - Vector2::repeat(constants.trace_size as i32 / 2);
    }
}

pub struct LightPlugin;
impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<LightConstants>()
            .init_resource::<LightParameters>()
            .add_systems(Startup, setup_fields)
            .add_systems(
                InitKernel,
                (init_wall_kernel, init_trace_kernel, init_accumulate_kernel),
            )
            .add_systems(Render, add_render(color).in_set(RenderPhase::Light));
    }
}
