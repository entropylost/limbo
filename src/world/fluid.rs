use sefirot::mapping::buffer::StaticDomain;
use sefirot_grid::dual::Facing;

use crate::prelude::*;
use crate::ui::debug::DebugCursor;

#[derive(Resource)]
pub struct FluidFields {
    pub ty: VField<u32, Cell>,
    pub next_ty: VField<u32, Cell>,
    pub velocity: VField<Vec2<f32>, Cell>,
    pub edge_velocity: VField<f32, Edge>,
    pub next_velocity: VField<Vec2<f32>, Cell>,
    pub solid: VField<bool, Cell>,
    _fields: FieldSet,
}

fn setup_fluids(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let fluid = FluidFields {
        ty: *fields.create_bind("fluid-ty", world.create_buffer(&device)),
        next_ty: *fields.create_bind("fluid-next-ty", world.create_buffer(&device)),
        velocity: *fields.create_bind("fluid-velocity", world.create_buffer(&device)),
        edge_velocity: *fields
            .create_bind("fluid-edge-velocity", world.dual.create_buffer(&device)),
        next_velocity: *fields.create_bind("fluid-next-velocity", world.create_buffer(&device)),
        solid: *fields.create_bind("fluid-solid", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(fluid);
}

#[kernel]
fn extract_edges(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if fluid.ty.expr(&cell) == 0 {
            return;
        }
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            let opposite = world.in_dir(&cell, dir);
            if fluid.ty.expr(&opposite) == 0 {
                *fluid.edge_velocity.var(&edge) =
                    Facing::from(dir).extract(fluid.velocity.expr(&cell));
            }
        }
    })
}

#[kernel]
fn extract_cells(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if fluid.ty.expr(&cell) == 0 {
            return;
        }
        let vel = Vec2::<f32>::var_zeroed();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *vel += fluid.edge_velocity.expr(&edge) * Facing::from(dir).as_vec_f32();
        }
        *fluid.velocity.var(&cell) = vel / 2.0;
    })
}
#[kernel]
fn reset_solid_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if !fluid.solid.expr(&cell) {
            return;
        }
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            *fluid.edge_velocity.var(&edge) = 0.0;
        }
    })
}

#[kernel]
fn blur_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &world.checkerboard(), &|cell| {
        if fluid.ty.expr(&cell) == 0 {
            return;
        }
        for [a, b] in [
            [GridDirection::Up, GridDirection::Down],
            [GridDirection::Right, GridDirection::Left],
        ] {
            let a = world.dual.in_dir(&cell, a);
            let b = world.dual.in_dir(&cell, b);
            let avg = (fluid.edge_velocity.expr(&a) + fluid.edge_velocity.expr(&b)) / 2.0;
            *fluid.edge_velocity.var(&a) = fluid.edge_velocity.expr(&a) * 0.9 + avg * 0.1;
            *fluid.edge_velocity.var(&b) = fluid.edge_velocity.expr(&b) * 0.9 + avg * 0.1;
        }
    })
}

#[kernel]
fn divergence_solve(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &world.checkerboard(), &|cell| {
        if fluid.ty.expr(&cell) == 0 {
            return;
        }
        let divergence = 0.0_f32.var();
        let solids = 0_u32.var();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&world.in_dir(&cell, dir)) {
                *divergence += fluid.edge_velocity.expr(&edge) * dir.signf();
                *solids += 1;
            }
        }
        *solids = max(solids, 1);
        let pressure = divergence / solids.cast_f32();
        for dir in GridDirection::iter_all() {
            let edge = world.dual.in_dir(&cell, dir);
            if !fluid.solid.expr(&world.in_dir(&cell, dir)) {
                *fluid.edge_velocity.var(&edge) += -pressure * dir.signf();
            }
        }
    })
}

#[kernel]
fn apply_gravity(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if fluid.ty.expr(&cell) != 0 || fluid.ty.expr(&world.in_dir(&cell, GridDirection::Up)) != 0
        {
            *fluid
                .edge_velocity
                .var(&world.dual.in_dir(&cell, GridDirection::Up)) -= 0.1;
        }
    })
}

#[kernel]
fn copy_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        *fluid.ty.var(&cell) = fluid.next_ty.expr(&cell);
        *fluid.velocity.var(&cell) = fluid.next_velocity.expr(&cell);
        *fluid.next_ty.var(&cell) = 0;
        *fluid.next_velocity.var(&cell) = Vec2::splat(0.0);
    })
}

#[tracked]
fn move_dir(fluid: &FluidFields, col: Element<Expr<u32>>, facing: Facing, single: bool) {
    let grid_point = |x: Expr<i32>| match facing {
        Facing::Horizontal => col.at(Vec2::expr(x, col.cast_i32())),
        Facing::Vertical => col.at(Vec2::expr(col.cast_i32(), x)),
    };
    let velocity = |cell: &Element<Expr<Vec2<i32>>>| {
        let v = match facing {
            Facing::Horizontal => fluid.velocity.expr(cell).x.cast_i32(),
            Facing::Vertical => fluid.velocity.expr(cell).y.cast_i32(),
        };
        if single && v < 0 {
            (-1).expr()
        } else if single && v > 0 {
            1.expr()
        } else {
            v
        }
    };
    // TODO: Can use union-find to find the nearest unoccupied cell.
    let lock = <[u32; 256]>::var([0; 256]);
    let vel = <[i32; 256]>::var([0; 256]);
    let reject_size = 0_u32.var();
    let reject = <[u32; 256]>::var([0; 256]);
    for i in 0..256_u32 {
        let i: Expr<u32> = i;
        if fluid.solid.expr(&grid_point(i.cast_i32())) {
            lock.write(i, 1);
        }
    }
    for i in 0..256_u32 {
        let i: Expr<u32> = i;
        let cell = grid_point(i.cast_i32());
        let ty = fluid.ty.expr(&cell);
        if ty == 0 {
            continue;
        }
        let v = velocity(&cell);
        let dst = (i.cast_i32() + v).clamp(0, 255).cast_u32();
        lock.write(dst, lock.read(dst) + 1);
    }
    for i in 0..256_u32 {
        let i: Expr<u32> = i;
        let cell = grid_point(i.cast_i32());
        let ty = fluid.ty.expr(&cell);
        if ty == 0 {
            continue;
        }
        let v = velocity(&cell);
        let dst = (i.cast_i32() + v).clamp(0, 255).cast_u32();
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
    for i in 0..256_u32 {
        let i: Expr<u32> = i;
        let cell = grid_point(i.cast_i32());
        let v = vel.read(i);
        let src = grid_point(i.cast_i32() - v);
        if lock.read(i) != 1 {
            continue;
        }
        *fluid.next_ty.var(&cell) = fluid.ty.expr(&src);
        *fluid.next_velocity.var(&cell) = fluid.velocity.expr(&src);
    }
}

#[kernel]
fn move_x_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(world.width()), &|col| {
        move_dir(&fluid, col, Facing::Horizontal, false);
    })
}
#[kernel]
fn move_y_kernel(device: Res<Device>, world: Res<World>, fluid: Res<FluidFields>) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(world.width()), &|col| {
        move_dir(&fluid, col, Facing::Vertical, false);
    })
}

#[kernel]
fn move_x_single_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(world.width()), &|col| {
        move_dir(&fluid, col, Facing::Horizontal, true);
    })
}
#[kernel]
fn move_y_single_kernel(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(world.width()), &|col| {
        move_dir(&fluid, col, Facing::Vertical, true);
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
fn cursor_kernel(device: Res<Device>, fluid: Res<FluidFields>) -> Kernel<fn(Vec2<i32>)> {
    Kernel::build(&device, &StaticDomain::<2>::new(8, 8), &|cell, cpos| {
        let pos = cpos + cell.cast_i32() - 4;
        let cell = cell.at(pos);
        *fluid.ty.var(&cell) = 1;
    })
}
#[kernel]
fn cursor_vel_kernel(
    device: Res<Device>,
    fluid: Res<FluidFields>,
) -> Kernel<fn(Vec2<i32>, Vec2<f32>)> {
    Kernel::build(
        &device,
        &StaticDomain::<2>::new(8, 8),
        &|cell, cpos, cvel| {
            let pos = cpos + cell.cast_i32() - 4;
            let cell = cell.at(pos);
            *fluid.velocity.var(&cell) = cvel;
        },
    )
}

fn update_fluids(
    mut parity: Local<bool>,
    cursor: Res<DebugCursor>,
    button: Res<ButtonInput<MouseButton>>,
) -> impl AsNodes {
    if button.pressed(MouseButton::Left) {
        cursor_kernel.dispatch_blocking(&Vec2::from(cursor.position.map(|x| x as i32)));
    }
    if button.pressed(MouseButton::Right) {
        cursor_vel_kernel.dispatch_blocking(
            &Vec2::from(cursor.position.map(|x| x as i32)),
            &Vec2::from(cursor.velocity / 60.0),
        );
    }
    *parity ^= true;
    let mv = if *parity {
        (
            move_x_kernel.dispatch(),
            copy_kernel.dispatch(),
            move_y_single_kernel.dispatch(),
            copy_kernel.dispatch(),
        )
            .chain()
    } else {
        (
            move_y_kernel.dispatch(),
            copy_kernel.dispatch(),
            move_x_single_kernel.dispatch(),
            copy_kernel.dispatch(),
        )
            .chain()
    };
    (
        extract_edges.dispatch(),
        apply_gravity.dispatch(),
        reset_solid_kernel.dispatch(),
        // blur_kernel.dispatch(),
        divergence_solve.dispatch(),
        // divergence_solve.dispatch(),
        // divergence_solve.dispatch(),
        extract_cells.dispatch(),
        mv,
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
                    init_copy_kernel,
                    init_apply_gravity,
                    init_move_x_kernel,
                    init_move_y_kernel,
                    init_move_x_single_kernel,
                    init_move_y_single_kernel,
                    init_cursor_kernel,
                    init_load_kernel,
                    init_extract_edges,
                    init_extract_cells,
                    init_reset_solid_kernel,
                    init_blur_kernel,
                    init_divergence_solve,
                ),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_fluids).in_set(UpdatePhase::Step),
            );
    }
}
