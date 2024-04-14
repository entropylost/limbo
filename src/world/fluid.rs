use sefirot::mapping::buffer::StaticDomain;
use sefirot_grid::dual::{DualGrid, Facing};
use sefirot_grid::GridDomain;

use crate::prelude::*;
use crate::ui::debug::DebugCursor;
use crate::utils::{rand, rand_f32};

#[derive(Resource)]
pub struct FlowFields {
    domain: GridDomain,
    dual: DualGrid,
    pub mass: VField<f32, Cell>,
    pub solid: VField<bool, Cell>,
    pub last_velocity: VField<f32, Edge>,
    pub velocity: VField<f32, Edge>,
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
    let domain = GridDomain::new_wrapping([0, 0], [256, 256]).with_morton(); // Needs to be pow 2, also uses 3x3 cells rn.
    let dual = domain.dual();

    let flow_mass = fields.create_bind("fluid-mass", domain.create_texture(&device));
    let flow_solid = *fields.create_bind("fluid-solid", domain.create_buffer(&device));
    let flow_last_velocity =
        fields.create_bind("fluid-last-velocity", dual.create_texture(&device));
    let flow_velocity = fields.create_bind("fluid-velocity", dual.create_texture(&device));

    let flow = FlowFields {
        domain,
        dual,
        mass: flow_mass,
        solid: flow_solid,
        last_velocity: flow_last_velocity,
        velocity: flow_velocity,
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

#[tracked]
fn bifilter(offset: Expr<Vec2<f32>>) -> Expr<f32> {
    max(1.0 - (offset / 3.0).abs(), 0.0).reduce_prod()
}

#[kernel]
fn p2g(device: Res<Device>, fluid: Res<FluidFields>, flow: Res<FlowFields>) -> Kernel<fn()> {
    Kernel::build(&device, &flow.domain, &|cell| {
        // let mass_pos = *cell * 3 + 1;
        // let mass = 0.0_f32.var();
        // for dx in -3..=3 {
        //     for dy in -3..=3 {
        //         let offset = Vec2::expr(dx, dy);
        //         let pos = mass_pos + offset;
        //         let pcell = cell.at(pos);
        //         let offset = offset.cast_f32();
        //         let weight = bifilter(offset);
        //         if fluid.ty.expr(&pcell) != 0 {
        //             *mass += weight;
        //         }
        //     }
        // }
        let xpos = (*cell * 3).cast_f32() + Vec2::expr(-0.5, 1.0);
        let xvel = 0.0_f32.var();
        // TODO: This shouldn't be necessary.
        let xweight = 0.00001_f32.var();
        // TODO: These are overly large.
        for dx in -4..=4 {
            for dy in -4..=4 {
                let pos = xpos.cast_i32() + Vec2::expr(dx, dy);
                let offset = pos.cast_f32() - xpos;
                let pcell = cell.at(pos);
                let weight = bifilter(offset);
                if fluid.ty.expr(&pcell) != 0 {
                    *xvel += fluid.velocity.expr(&pcell).x * weight;
                    *xweight += weight;
                }
            }
        }
        let ypos = (*cell * 3).cast_f32() + Vec2::expr(-0.5, 1.0);
        let yvel = 0.0_f32.var();
        let yweight = 0.00001_f32.var();
        // TODO: These are overly large.
        for dx in -4..=4 {
            for dy in -4..=4 {
                let pos = ypos.cast_i32() + Vec2::expr(dx, dy);
                let offset = pos.cast_f32() - ypos;
                let pcell = cell.at(pos);
                let weight = bifilter(offset);
                if fluid.ty.expr(&pcell) != 0 {
                    *yvel += fluid.velocity.expr(&pcell).y * weight;
                    *yweight += weight;
                }
            }
        }
        *flow
            .velocity
            .var(&flow.dual.in_dir(&cell, GridDirection::Left)) = xvel / xweight;
        *flow
            .velocity
            .var(&flow.dual.in_dir(&cell, GridDirection::Down)) = yvel / yweight;
        let mass = 0.0_f32.var();
        let solid = false.var();
        for dx in 0..3 {
            for dy in 0..3 {
                let pcell = cell.at(*cell * 3 + Vec2::expr(dx, dy));
                if fluid.ty.expr(&pcell) != 0 {
                    *mass += 1.0;
                }
                if fluid.solid.expr(&pcell) {
                    *solid = true;
                }
            }
        }
        // TODO: This isn't properly blurred so cannot be used for averaging. Is necessary to decide which grid cells to run though.
        *flow.mass.var(&cell) = mass;
        *flow.solid.var(&cell) = solid;
    })
}

#[kernel]
fn divergence_solve(
    device: Res<Device>,
    fluid: Res<FluidFields>,
    flow: Res<FlowFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &flow.domain.checkerboard(), &|cell| {
        if flow.mass.expr(&cell) == 0.0 || flow.solid.expr(&cell) {
            for dir in GridDirection::iter_all() {
                let edge = flow.dual.in_dir(&cell, dir);
                *flow.velocity.var(&edge) = 0.0;
            }
            return;
        }
        let divergence = 0.0_f32.var();
        let solids = 0_u32.var();
        for dir in GridDirection::iter_all() {
            let edge = flow.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&flow.domain.in_dir(&cell, dir)) {
                *divergence += flow.velocity.expr(&edge) * dir.signf();
                *solids += 1;
            }
        }
        *solids = max(solids, 1);
        let pressure = 1.9 * divergence / solids.cast_f32();
        for dir in GridDirection::iter_all() {
            let edge = flow.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&flow.domain.in_dir(&cell, dir)) {
                *flow.velocity.var(&edge) += -pressure * dir.signf();
            }
        }
    })
}

#[kernel]
fn g2p(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
    flow: Res<FlowFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        //  TODO: +FLIP
        if fluid.ty.expr(&cell) == 0 {
            return;
        }
        let x1cell = cell.at((*cell - Vec2::expr(0, 1)) / 3);
        let x1loff = (*x1cell * 3).cast_f32() + Vec2::new(-0.5, 1.0) - cell.cast_f32();
        let x2cell = cell.at(*x1cell + Vec2::expr(0, 1));
        let xvel = flow
            .velocity
            .expr(&flow.dual.in_dir(&x1cell, GridDirection::Left))
            * bifilter(x1loff)
            + flow
                .velocity
                .expr(&flow.dual.in_dir(&x1cell, GridDirection::Right))
                * bifilter(x1loff + Vec2::expr(3.0, 0.0))
            + flow
                .velocity
                .expr(&flow.dual.in_dir(&x2cell, GridDirection::Left))
                * bifilter(x1loff + Vec2::expr(0.0, 3.0))
            + flow
                .velocity
                .expr(&flow.dual.in_dir(&x2cell, GridDirection::Right))
                * bifilter(x1loff + Vec2::expr(3.0, 3.0));
        let y1cell = cell.at((*cell - Vec2::expr(1, 0)) / 3);
        let y1doff = (*y1cell * 3).cast_f32() + Vec2::new(1.0, -0.5) - cell.cast_f32();
        let y2cell = cell.at(*y1cell + Vec2::expr(1, 0));
        let yvel = flow
            .velocity
            .expr(&flow.dual.in_dir(&y1cell, GridDirection::Down))
            * bifilter(y1doff)
            + flow
                .velocity
                .expr(&flow.dual.in_dir(&y1cell, GridDirection::Up))
                * bifilter(y1doff + Vec2::expr(0.0, 3.0))
            + flow
                .velocity
                .expr(&flow.dual.in_dir(&y2cell, GridDirection::Down))
                * bifilter(y1doff + Vec2::expr(3.0, 0.0))
            + flow
                .velocity
                .expr(&flow.dual.in_dir(&y2cell, GridDirection::Up))
                * bifilter(y1doff + Vec2::expr(3.0, 3.0));
        *fluid.velocity.var(&cell) = Vec2::expr(xvel, yvel - 0.01);
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
        if cell.y < 60 || cell.x < 39 {
            *fluid.solid.var(&cell) = true;
            // *fluid.mass.var(&cell) = 1.0;
        }
    })
}

#[kernel]
fn cursor_kernel(device: Res<Device>, fluid: Res<FluidFields>) -> Kernel<fn(Vec2<i32>)> {
    Kernel::build(&device, &StaticDomain::<2>::new(8, 8), &|cell, cpos| {
        let pos = cpos + cell.cast_i32() - 4;
        let cell = cell.at(pos);
        *fluid.ty.var(&cell) = 1;
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
        p2g.dispatch(),
        divergence_solve.dispatch(),
        g2p.dispatch(),
        velocity_kernel.dispatch(&*t),
        mv2,
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
                    init_copy_fluid_kernel,
                    init_wall_kernel,
                    init_move_x_kernel,
                    init_move_y_kernel,
                    init_cursor_kernel,
                    init_load_kernel,
                    init_paint_kernel,
                    init_premove_kernel,
                    init_brownian_motion_kernel,
                    init_velocity_kernel,
                    init_average_velocity_kernel,
                    init_divergence_solve,
                    init_p2g,
                    init_g2p,
                ),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_fluids).in_set(UpdatePhase::Step),
            );
    }
}
