use sefirot::mapping::buffer::StaticDomain;
use sefirot_grid::dual::Facing;

use crate::prelude::*;
use crate::ui::debug::DebugCursor;
use crate::utils::{rand, rand_f32};

#[derive(Resource)]
pub struct FlowFields {
    pub mass: VField<f32, Cell>,
    pub next_mass: AField<f32, Cell>,
    pub velocity: VField<f32, Edge>,
    pub next_momentum: AField<f32, Edge>,
}

#[derive(Resource)]
pub struct FluidFields {
    pub ty: VField<u32, Cell>,
    pub next_ty: VField<u32, Cell>,
    pub velocity: VField<Vec2<f32>, Cell>,
    pub next_velocity: VField<Vec2<f32>, Cell>,
    pub delta: VField<Vec2<i32>, Cell>,
    pub movement: VField<Vec2<i32>, Cell>,
    pub solid: VField<bool, Cell>,
    pub avg_velocity: VField<Vec2<f32>, Cell>,
    pub next_avg_velocity: VField<Vec2<f32>, Cell>,
    _fields: FieldSet,
}

fn setup_fluids(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let flow = FlowFields {
        mass: fields.create_bind("fluid-mass", world.create_texture(&device)),
        next_mass: fields.create_bind("fluid-next-mass", world.create_buffer(&device)),
        velocity: fields.create_bind("fluid-velocity", world.dual.create_texture(&device)),
        next_momentum: fields.create_bind("fluid-next-momentum", world.dual.create_buffer(&device)),
    };
    commands.insert_resource(flow);

    let fluid = FluidFields {
        ty: *fields.create_bind("fluid-ty", world.create_buffer(&device)),
        next_ty: *fields.create_bind("fluid-next-ty", world.create_buffer(&device)),
        velocity: *fields.create_bind("fluid-velocity", world.create_buffer(&device)),
        next_velocity: *fields.create_bind("fluid-next-velocity", world.create_buffer(&device)),
        delta: *fields.create_bind("fluid-delta", world.create_buffer(&device)),
        movement: *fields.create_bind("fluid-movement", world.create_buffer(&device)),
        solid: *fields.create_bind("fluid-solid", world.create_buffer(&device)),
        avg_velocity: *fields.create_bind("fluid-adv-velocity", world.create_buffer(&device)),
        next_avg_velocity: *fields
            .create_bind("fluid-next-adv-velocity", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(fluid);
}

#[kernel]
fn premove_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *fluid.next_velocity.var(&cell) = fluid.velocity.expr(&cell);
        *fluid.next_avg_velocity.var(&cell) = fluid.avg_velocity.expr(&cell);
    })
}

#[kernel]
fn extract_edges(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
    flow: Res<FlowFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if fluid.ty.expr(&cell) == 0 {
            return;
        }
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            let opposite = world.in_dir(&cell, dir);
            if fluid.ty.expr(&opposite) == 0 && !fluid.solid.expr(&opposite) {
                *flow.velocity.var(&edge) = Facing::from(dir).extract(fluid.velocity.expr(&cell));
            }
        }
        *flow.mass.var(&cell) += 0.01;
    })
}

#[kernel]
fn extract_cells(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
    flow: Res<FlowFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let vel = Vec2::<f32>::var_zeroed();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *vel += flow.velocity.expr(&edge) * Facing::from(dir).as_vec_f32();
        }
        *vel /= 2.0;
        if fluid.ty.expr(&cell) != 0 {
            let f = min(flow.mass.expr(&cell), 1.0);
            *fluid.velocity.var(&cell) = fluid.velocity.expr(&cell) * (1.0 - f) + vel * f;
        }
    })
}
#[kernel]
fn divergence_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
    flow: Res<FlowFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.checkerboard(), &|cell| {
        if fluid.solid.expr(&cell) {
            for dir in GridDirection::iter_all() {
                let edge = world.dual.in_dir(&cell, dir);
                *flow.velocity.var(&edge) = 0.0;
            }
            return;
        }
        let divergence = 0.0_f32.var();
        let solids = 0_u32.var();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&world.in_dir(&cell, dir)) {
                *divergence += flow.velocity.expr(&edge) * dir.signf();
                *solids += 1;
            }
        }
        *solids = max(solids, 1);
        let pressure = 0.1 * divergence / solids.cast_f32()
            - 0.1 * max(flow.mass.expr(&cell) - 1.0, 0.0) * 4.0 / solids.cast_f32();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&world.in_dir(&cell, dir)) {
                *flow.velocity.var(&edge) += -pressure * dir.signf();
            }
        }
    })
}

#[kernel]
fn velocity_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn(u32)> {
    // Might be worth splitting the positive and negative movements.
    Kernel::build(&device, &**world, &|cell, t| {
        let cutoff = Vec2::expr(
            rand_f32(cell.cast_u32(), t, 0),
            rand_f32(cell.cast_u32(), t, 1),
        );
        if fluid.ty.expr(&cell) != 0 {
            let vel = fluid.velocity.expr(&cell) * 1.5;
            let ivel = vel.round().cast_i32();
            let fvel = vel - ivel.cast_f32();
            let fvel_sign = fvel.signum().cast_i32();
            let mask = fvel.abs() * 2.0 > cutoff;
            *fluid.delta.var(&cell) = ivel + mask.cast_i32() * fvel_sign;
        }
    })
}

#[kernel]
fn brownian_motion_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn(u32)> {
    Kernel::build(&device, &**world, &|cell, t| {
        let dir = rand(cell.cast_u32(), t, 0) % 4;
        if fluid.ty.expr(&cell) != 0 {
            *fluid.delta.var(&cell) = [Vec2::new(1_i32, 0), Vec2::new(0, 1_i32)]
                .expr()
                .read(dir % 2)
                * (2 * (dir.cast_i32() / 2) - 1);
        }
    })
}

#[kernel]
fn average_velocity_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if fluid.ty.expr(&cell) != 0 {
            *fluid.velocity.var(&cell) =
                0.99 * fluid.velocity.expr(&cell) + 0.01 * fluid.delta.expr(&cell).cast_f32();
            // + Vec2::new(0.0, -0.01);
        }
    })
}

#[kernel]
fn copy_fluid_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *fluid.ty.var(&cell) = fluid.next_ty.expr(&cell);
        if fluid.ty.expr(&cell) != 0 {
            let delta = fluid.movement.expr(&cell);
            let src = cell.at(*cell - delta);
            *fluid.velocity.var(&cell) = fluid.next_velocity.expr(&src);
            *fluid.avg_velocity.var(&cell) = fluid.next_avg_velocity.expr(&src);
        } else {
            *fluid.velocity.var(&cell) = Vec2::splat(0.0);
            *fluid.avg_velocity.var(&cell) = Vec2::splat(0.0);
        }
        *fluid.next_ty.var(&cell) = 0;
    })
}

#[kernel]
fn clear_kernel(device: Res<Device>, world: Res<World>, flow: Res<FlowFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *flow.next_mass.var(&cell) = 0.0;
        for dir in [GridDirection::Right, GridDirection::Up] {
            let edge = world.dual.in_dir(&cell, dir);
            *flow.next_momentum.var(&edge) = 0.0;
        }
    })
}

#[kernel]
fn copy_flow_kernel(
    device: Res<Device>,
    world: Res<World>,
    flow: Res<FlowFields>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *flow.mass.var(&cell) = flow.next_mass.expr(&cell)
            * if fluid.ty.expr(&cell) == 0 {
                0.99.expr()
            } else {
                1.0_f32.expr()
            };
        for dir in [GridDirection::Right, GridDirection::Up] {
            let edge = world.dual.in_dir(&cell, dir);
            let opposite = world.in_dir(&cell, dir);
            let weight = max(
                flow.next_mass.expr(&cell) + flow.next_mass.expr(&opposite),
                0.0001,
            );
            if dir == GridDirection::Up {
                *flow.velocity.var(&edge) = flow.next_momentum.expr(&edge) / weight - 0.005_f32;
            } else {
                *flow.velocity.var(&edge) = flow.next_momentum.expr(&edge) / weight;
            }
        }
    })
}

#[kernel]
fn advect_kernel(device: Res<Device>, world: Res<World>, flow: Res<FlowFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let vel_start_x = flow
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Left));
        let vel_end_x = flow
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Right));
        let vel_start_y = flow
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Down));
        let vel_end_y = flow
            .velocity
            .expr(&world.dual.in_dir(&cell, GridDirection::Up));

        let a = Vec2::expr(vel_start_x, vel_start_y);
        let b = Vec2::expr(vel_end_x, vel_end_y) + 1.0;
        let start = min(a, b);
        let end = max(a, b);
        let density = flow.mass.expr(&cell) * 1.0 / max((end - start).reduce_prod(), 0.00001);
        if density < 0.0001 {
            return;
        }
        let density = flow.mass.expr(&cell) * 1.0 / max((end - start).reduce_prod(), 0.00001);
        for i in start.x.floor().cast_i32()..end.x.ceil().cast_i32() {
            for j in start.y.floor().cast_i32()..end.y.ceil().cast_i32() {
                let offset = Vec2::expr(i, j);
                let dst = cell.at(offset + *cell);
                let offset = offset.cast_f32();
                if !world.contains(&dst) {
                    continue;
                }
                let intersection = min(end, offset + 1.0) - max(start, offset);
                let weight = density * intersection.reduce_prod();
                flow.next_mass.atomic(&dst).fetch_add(weight);
                // TODO: These break.
                let dst_x_start_inv = (offset.x - a.x) / (b.x - a.x);
                let dst_y_start_inv = (offset.y - a.y) / (b.y - a.y);
                let dst_x_end_inv = (offset.x + 1.0 - a.x) / (b.x - a.x);
                let dst_y_end_inv = (offset.y + 1.0 - a.y) / (b.y - a.y);

                flow.next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Left))
                    .fetch_add(
                        lerp(dst_x_start_inv.clamp(0.0, 1.0), vel_start_x, vel_end_x) * weight,
                    );
                flow.next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Right))
                    .fetch_add(
                        lerp(dst_x_end_inv.clamp(0.0, 1.0), vel_start_x, vel_end_x) * weight,
                    );
                flow.next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Down))
                    .fetch_add(
                        lerp(dst_y_start_inv.clamp(0.0, 1.0), vel_start_y, vel_end_y) * weight,
                    );
                flow.next_momentum
                    .atomic(&world.dual.in_dir(&dst, GridDirection::Up))
                    .fetch_add(
                        lerp(dst_y_end_inv.clamp(0.0, 1.0), vel_start_y, vel_end_y) * weight,
                    );
            }
        }
    })
}

#[tracked]
fn move_dir(fluid: &FluidFields, col: Element<Expr<u32>>, facing: Facing) {
    let grid_point = |x: Expr<i32>| match facing {
        Facing::Horizontal => col.at(Vec2::expr(x, col.cast_i32())),
        Facing::Vertical => col.at(Vec2::expr(col.cast_i32(), x)),
    };
    let velocity = |cell: &Element<Expr<Vec2<i32>>>| match facing {
        Facing::Horizontal => fluid.delta.expr(cell).x,
        Facing::Vertical => fluid.delta.expr(cell).y,
    };
    // TODO: Can use union-find to find the nearest unoccupied cell.
    let lock = <[u32; 512]>::var([0; 512]);
    let vel = <[i32; 512]>::var([0; 512]);
    let reject_size = 0_u32.var();
    let reject = <[u32; 512]>::var([0; 512]);
    for i in 0..512_u32 {
        let i: Expr<u32> = i;
        if fluid.solid.expr(&grid_point(i.cast_i32())) {
            lock.write(i, 1);
        }
    }
    for i in 0..512_u32 {
        let i: Expr<u32> = i;
        let cell = grid_point(i.cast_i32());
        let ty = fluid.ty.expr(&cell);
        if ty == 0 {
            continue;
        }
        let v = velocity(&cell);
        let dst = (i.cast_i32() + v).rem_euclid(512).cast_u32();
        lock.write(dst, lock.read(dst) + 1);
    }
    for i in 0..512_u32 {
        let i: Expr<u32> = i;
        let cell = grid_point(i.cast_i32());
        let ty = fluid.ty.expr(&cell);
        if ty == 0 {
            continue;
        }
        let v = velocity(&cell);
        let dst = (i.cast_i32() + v).rem_euclid(512).cast_u32();
        if lock.read(dst) == 1 {
            vel.write(dst, (dst - i).cast_i32());
        } else {
            reject.write(reject_size, i);
            *reject_size += 1;
        }
    }
    while reject_size > 0 {
        let i = reject.read(reject_size - 1);
        *reject_size -= 1;
        let s = vel.read(i);
        lock.write(i, 1);
        if s != 0 {
            let j = i.cast_i32() - s;
            vel.write(i, 0);
            reject.write(reject_size, j.cast_u32());
            *reject_size += 1;
        }
    }
    for i in 0..512_u32 {
        let i: Expr<u32> = i;
        let cell = grid_point(i.cast_i32());
        let v = vel.read(i);
        let src = grid_point(i.cast_i32() - v);
        if lock.read(i) != 1 {
            continue;
        }

        *fluid.next_ty.var(&cell) = fluid.ty.expr(&src);
        *fluid.movement.var(&cell) = match facing {
            Facing::Horizontal => Vec2::expr(v, 0),
            Facing::Vertical => Vec2::expr(0, v),
        };
    }
}
#[kernel]
fn move_x_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(world.height()), &|col| {
        move_dir(&fluid, col, Facing::Horizontal);
    })
}
#[kernel]
fn move_y_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(world.width()), &|col| {
        move_dir(&fluid, col, Facing::Vertical);
    })
}

#[kernel(run)]
fn load_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if cell.y < 60 || cell.x < 40 {
            *fluid.solid.var(&cell) = true;
            // *fluid.mass.var(&cell) = 1.0;
        }
    })
}

#[kernel]
fn cursor_kernel(
    device: Res<Device>,
    fluid: Res<FluidFields>,
    flow: Res<FlowFields>,
) -> Kernel<fn(Vec2<i32>)> {
    Kernel::build(&device, &StaticDomain::<2>::new(8, 8), &|cell, cpos| {
        let pos = cpos + cell.cast_i32() - 4;
        let cell = cell.at(pos);
        *fluid.ty.var(&cell) = 1;
        *flow.mass.var(&cell) = 1.0;
    })
}
#[kernel]
fn paint_kernel(device: Res<Device>, fluid: Res<FluidFields>) -> Kernel<fn(Vec2<i32>)> {
    Kernel::build(&device, &StaticDomain::<2>::new(8, 8), &|cell, cpos| {
        let pos = cpos + cell.cast_i32() - 4;
        let cell = cell.at(pos);
        if fluid.ty.expr(&cell) == 1 {
            *fluid.ty.var(&cell) = 2;
        }
    })
}

#[kernel]
fn cursor_vel_kernel(
    device: Res<Device>,
    fluid: Res<FluidFields>,
) -> Kernel<fn(Vec2<i32>, Vec2<f32>)> {
    Kernel::build(
        &device,
        &StaticDomain::<2>::new(32, 32),
        &|cell, cpos, cvel| {
            let pos = cpos + cell.cast_i32() - 16;
            let cell = cell.at(pos);
            *fluid.velocity.var(&cell) = cvel;
        },
    )
}

#[kernel]
fn wall_kernel(device: Res<Device>, fluid: Res<FluidFields>) -> Kernel<fn(Vec2<i32>, bool)> {
    Kernel::build(
        &device,
        &StaticDomain::<2>::new(8, 8),
        &|cell, cpos, wall| {
            let pos = cpos + cell.cast_i32() - 4;
            let cell = cell.at(pos);
            *fluid.solid.var(&cell) = wall;
        },
    )
}

fn update_fluids(
    mut parity: Local<bool>,
    mut t: Local<u32>,
    cursor: Res<DebugCursor>,
    button: Res<ButtonInput<MouseButton>>,
) -> impl AsNodes {
    if cursor.on_world {
        if button.pressed(MouseButton::Left) {
            cursor_kernel.dispatch_blocking(&Vec2::from(cursor.position.map(|x| x as i32)));
        }
        if button.pressed(MouseButton::Middle) {
            wall_kernel.dispatch_blocking(&Vec2::from(cursor.position.map(|x| x as i32)), &true);
        }
        if button.pressed(MouseButton::Right) {
            wall_kernel.dispatch_blocking(&Vec2::from(cursor.position.map(|x| x as i32)), &false);
        }
    }
    // cursor_vel_kernel.dispatch_blocking(
    //     &Vec2::from(cursor.position.map(|x| x as i32)),
    //     &Vec2::from(cursor.velocity / 60.0),
    // );
    *parity ^= true;
    *t += 1;
    let mv1 = if *parity {
        (
            premove_kernel.dispatch(),
            move_y_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
            premove_kernel.dispatch(),
            move_x_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
        )
            .chain()
    } else {
        (
            premove_kernel.dispatch(),
            move_x_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
            premove_kernel.dispatch(),
            move_y_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
        )
            .chain()
    };
    let mv2 = if *parity {
        (
            premove_kernel.dispatch(),
            move_y_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
            premove_kernel.dispatch(),
            move_x_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
        )
            .chain()
    } else {
        (
            premove_kernel.dispatch(),
            move_x_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
            premove_kernel.dispatch(),
            move_y_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
        )
            .chain()
    };
    (
        brownian_motion_kernel.dispatch(&*t),
        mv1,
        average_velocity_kernel.dispatch(),
        extract_edges.dispatch(),
        velocity_kernel.dispatch(&*t),
        mv2,
        advect_kernel.dispatch(),
        copy_flow_kernel.dispatch(),
        clear_kernel.dispatch(),
        divergence_kernel.dispatch(),
        divergence_kernel.dispatch(),
        extract_cells.dispatch(),
    )
        .chain()
}

pub struct FluidPlugin;
impl Plugin for FluidPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_fluids)
            .add_systems(
                InitKernel,
                (
                    init_cursor_vel_kernel,
                    init_copy_flow_kernel,
                    init_copy_fluid_kernel,
                    init_wall_kernel,
                    init_move_x_kernel,
                    init_move_y_kernel,
                    init_cursor_kernel,
                    init_load_kernel,
                    init_extract_edges,
                    init_extract_cells,
                    init_advect_kernel,
                    init_clear_kernel,
                    init_paint_kernel,
                    init_divergence_kernel,
                    init_premove_kernel,
                    init_brownian_motion_kernel,
                    init_velocity_kernel,
                    init_average_velocity_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_fluids).in_set(UpdatePhase::Step),
            );
    }
}
