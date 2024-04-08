use crate::prelude::*;

#[derive(Resource)]
pub struct FluidFields {
    pub mass: VField<f32, Cell>,
    pub next_mass: AField<f32, Cell>,
    pub velocity: VField<f32, Edge>,
    pub next_momentum: AField<f32, Edge>,
    pub solid: VField<bool, Cell>,
    pub ty: VField<u32, Cell>,
    pub next_ty: AField<u32, Cell>,
    pub lock: VField<bool, Cell>,
    _fields: FieldSet,
}

fn setup_fluids(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let fluid = FluidFields {
        mass: fields.create_bind("fluid-mass", world.create_texture(&device)),
        next_mass: fields.create_bind("fluid-next-mass", world.create_buffer(&device)),
        velocity: fields.create_bind("fluid-velocity", world.dual.create_texture(&device)),
        next_momentum: fields.create_bind("fluid-next-momentum", world.dual.create_buffer(&device)),
        solid: *fields.create_bind("fluid-solid", world.create_buffer(&device)),
        ty: *fields.create_bind("fluid-ty", world.create_buffer(&device)),
        next_ty: fields.create_bind("fluid-next-ty", world.create_buffer(&device)),
        lock: *fields.create_bind("fluid-lock", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(fluid);
}

#[kernel]
fn divergence_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.checkerboard(), &|cell| {
        if fluid.solid.expr(&cell) {
            for dir in GridDirection::iter_all() {
                let edge = world.dual.in_dir(&cell, dir);
                *fluid.velocity.var(&edge) = 0.0;
            }
            return;
        }
        let divergence = 0.0_f32.var();
        let solids = 0_u32.var();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&world.in_dir(&cell, dir)) {
                *divergence += fluid.velocity.expr(&edge) * dir.signf();
                *solids += 1;
            }
        }
        *solids = max(solids, 1);
        let pressure = 0.1 * divergence / solids.cast_f32()
            - 0.1 * max(fluid.mass.expr(&cell) - 1.0, 0.0) * 4.0 / solids.cast_f32();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&world.in_dir(&cell, dir)) {
                *fluid.velocity.var(&edge) += -pressure * dir.signf();
            }
        }
    })
}

#[kernel]
fn copy_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *fluid.mass.var(&cell) = fluid.next_mass.expr(&cell);
        *fluid.ty.var(&cell) = fluid.next_ty.expr(&cell);
        for dir in [GridDirection::Right, GridDirection::Up] {
            let edge = world.dual.in_dir(&cell, dir);
            let opposite = world.in_dir(&cell, dir);
            let weight = max(
                fluid.next_mass.expr(&cell) + fluid.next_mass.expr(&opposite),
                0.0001,
            );
            if dir == GridDirection::Up {
                *fluid.velocity.var(&edge) =
                    fluid.next_momentum.expr(&edge) / weight - 0.005_f32 / 40.0;
            } else {
                *fluid.velocity.var(&edge) = fluid.next_momentum.expr(&edge) / weight;
            }
        }
    })
}

#[kernel]
fn clear_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *fluid.next_mass.var(&cell) = 0.0;
        *fluid.next_ty.var(&cell) = 0;
        *fluid.lock.var(&cell) = false;
        for dir in [GridDirection::Right, GridDirection::Up] {
            let edge = world.dual.in_dir(&cell, dir);
            *fluid.next_momentum.var(&edge) = 0.0;
        }
    })
}

#[kernel]
fn lock_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let vel_start_x = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Left));
        let vel_end_x = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Right));
        let vel_start_y = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Down));
        let vel_end_y = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Up));

        let a = Vec2::expr(vel_start_x, vel_start_y);
        let b = Vec2::expr(vel_end_x, vel_end_y) + 1.0;
        let start = min(a, b);
        let end = max(a, b);
        let density = fluid.mass.expr(&cell) * 1.0 / max((end - start).reduce_prod(), 0.00001);
        if density < 0.0001 {
            return;
        }
        let ty = fluid.ty.expr(&cell);
        for i in start.x.floor().cast_i32()..end.x.ceil().cast_i32() {
            for j in start.y.floor().cast_i32()..end.y.ceil().cast_i32() {
                let offset = Vec2::expr(i, j);
                let dst = cell.at(offset + *cell);
                if !world.contains(&dst) {
                    continue;
                }
                let x = fluid.next_ty.atomic(&dst).compare_exchange(0, ty);
                if x != 0 && x != ty {
                    *fluid.lock.var(&dst) = true;
                }
            }
        }
    })
}

#[kernel]
fn advect_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let vel_start_x = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Left));
        let vel_end_x = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Right));
        let vel_start_y = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Down));
        let vel_end_y = fluid
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Up));

        let a = Vec2::expr(vel_start_x, vel_start_y);
        let b = Vec2::expr(vel_end_x, vel_end_y) + 1.0;
        let start = min(a, b);
        let end = max(a, b);
        let density = fluid.mass.expr(&cell) * 1.0 / max((end - start).reduce_prod(), 0.00001);
        if density < 0.0001 {
            return;
        }
        let overflow = 0.0_f32.var();
        for i in start.x.floor().cast_i32()..end.x.ceil().cast_i32() {
            for j in start.y.floor().cast_i32()..end.y.ceil().cast_i32() {
                let offset = Vec2::expr(i, j);
                let dst = cell.at(offset + *cell);
                let offset = offset.cast_f32();
                if !world.contains(&dst) {
                    continue;
                }
                let intersection = min(end, offset + 1.0) - max(start, offset);
                if fluid.lock.expr(&dst) {
                    *overflow += intersection.reduce_prod();
                }
            }
        }
        let density =
            fluid.mass.expr(&cell) * 1.0 / max((end - start).reduce_prod() - overflow, 0.00001);
        for i in start.x.floor().cast_i32()..end.x.ceil().cast_i32() {
            for j in start.y.floor().cast_i32()..end.y.ceil().cast_i32() {
                let offset = Vec2::expr(i, j);
                let dst = cell.at(offset + *cell);
                let offset = offset.cast_f32();
                if !world.contains(&dst) {
                    continue;
                }
                let lock = fluid.lock.expr(&dst);
                if lock {
                    continue;
                }
                let intersection = min(end, offset + 1.0) - max(start, offset);
                let weight = density * intersection.reduce_prod();
                fluid.next_mass.atomic(&dst).fetch_add(weight);
                // TODO: These break.
                let dst_x_start_inv = (offset.x - a.x) / (b.x - a.x);
                let dst_y_start_inv = (offset.y - a.y) / (b.y - a.y);
                let dst_x_end_inv = (offset.x + 1.0 - a.x) / (b.x - a.x);
                let dst_y_end_inv = (offset.y + 1.0 - a.y) / (b.y - a.y);

                fluid
                    .next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Left))
                    .fetch_add(
                        lerp(dst_x_start_inv.clamp(0.0, 1.0), vel_start_x, vel_end_x) * weight,
                    );
                fluid
                    .next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Right))
                    .fetch_add(
                        lerp(dst_x_end_inv.clamp(0.0, 1.0), vel_start_x, vel_end_x) * weight,
                    );
                fluid
                    .next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Down))
                    .fetch_add(
                        lerp(dst_y_start_inv.clamp(0.0, 1.0), vel_start_y, vel_end_y) * weight,
                    );
                fluid
                    .next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Up))
                    .fetch_add(
                        lerp(dst_y_end_inv.clamp(0.0, 1.0), vel_start_y, vel_end_y) * weight,
                    );
            }
        }
    })
}

#[kernel(run)]
fn load_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if cell.y >= 60 && cell.x >= 40 && cell.x <= 100 && cell.y < 120 {
            *fluid.mass.var(&cell) = 1.0;
            *fluid.ty.var(&cell) = 1;

            if cell.x < 128 {
                *fluid.ty.var(&cell) = 1;
            }
        }
        if cell.y < 60 || cell.x < 40 {
            *fluid.solid.var(&cell) = true;
            // *fluid.mass.var(&cell) = 1.0;
        }
    })
}

fn update_fluids() -> impl AsNodes {
    let iters = 10;
    (0..iters)
        .map(|_| {
            (
                lock_kernel.dispatch(),
                advect_kernel.dispatch(),
                copy_kernel.dispatch(),
                clear_kernel.dispatch(),
                divergence_kernel.dispatch(),
                divergence_kernel.dispatch(),
                divergence_kernel.dispatch(),
                divergence_kernel.dispatch(),
            )
                .chain()
        })
        .collect::<Vec<_>>()
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
                    init_clear_kernel,
                    init_copy_kernel,
                    init_divergence_kernel,
                    init_lock_kernel,
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
