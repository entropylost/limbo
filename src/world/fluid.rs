use sefirot::mapping::buffer::StaticDomain;
use sefirot_grid::dual::Facing;

use super::direction::Direction;
use crate::prelude::*;
use crate::ui::debug::DebugCursor;

#[derive(Resource)]
pub struct FluidFields {
    pub ty: VField<u32, Cell>,
    pub next_ty: VField<u32, Cell>,
    pub velocity: VField<Vec2<f32>, Cell>,
    pub next_velocity: VField<Vec2<f32>, Cell>,
    pub solid: VField<bool, Cell>,
    pub pressure: VField<f32, Cell>,
    _fields: FieldSet,
}

fn setup_fluids(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let fluid = FluidFields {
        ty: *fields.create_bind("fluid-ty", world.create_buffer(&device)),
        next_ty: *fields.create_bind("fluid-next-ty", world.create_buffer(&device)),
        velocity: *fields.create_bind("fluid-velocity", world.create_buffer(&device)),
        next_velocity: *fields.create_bind("fluid-next-velocity", world.create_buffer(&device)),
        solid: *fields.create_bind("fluid-solid", world.create_buffer(&device)),
        pressure: *fields.create_bind("fluid-pressure", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(fluid);
}

#[kernel]
fn compute_velocity(
    device: Res<Device>,
    world: Res<World>,
    fluid: Res<FluidFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        if fluid.ty.expr(&cell) == 0 {
            *fluid.pressure.var(&cell) *= 0.99;
            return;
        }
        let normal = Vec2::<f32>::var_zeroed();
        for dir in Direction::iter_all() {
            let other = cell.at(*cell + dir.as_vec());

            if fluid.ty.expr(&other) != 0 {
                *normal += dir.as_vec_f32() * fluid.pressure.expr(&other);
            }
        }
        let vel = -(normal + Vec2::new(0.0, -0.1)).normalize();
        *fluid.velocity.var(&cell) = vel + fluid.velocity.expr(&cell) * 0.0001;
        *fluid.pressure.var(&cell) += 0.01;
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
        if single {
            match facing {
                Facing::Horizontal => fluid.velocity.expr(cell).x.signum().cast_i32(),
                Facing::Vertical => fluid.velocity.expr(cell).y.signum().cast_i32(),
            }
        } else {
            match facing {
                Facing::Horizontal => fluid.velocity.expr(cell).x.round().cast_i32(),
                Facing::Vertical => fluid.velocity.expr(cell).y.round().cast_i32(),
            }
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
        let dst = (i.cast_i32() + v).rem_euclid(256).cast_u32();
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
        let dst = (i.cast_i32() + v).rem_euclid(256).cast_u32();
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
        if cell.x == cell.y && cell.x == 128 {
            *fluid.ty.var(&cell) = 1;
            *fluid.velocity.var(&cell) = Vec2::splat(1.0);
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
    let mv = if *parity {
        (
            move_x_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
            // move_y_single_kernel.dispatch(),
            // copy_fluid_kernel.dispatch(),
        )
            .chain()
    } else {
        (
            move_y_kernel.dispatch(),
            copy_fluid_kernel.dispatch(),
            // move_x_single_kernel.dispatch(),
            // copy_fluid_kernel.dispatch(),
        )
            .chain()
    };
    (compute_velocity.dispatch(), mv).chain()
}

pub struct FluidPlugin;
impl Plugin for FluidPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_fluids)
            .add_systems(
                InitKernel,
                (
                    init_cursor_vel_kernel,
                    init_compute_velocity,
                    init_copy_fluid_kernel,
                    init_wall_kernel,
                    init_move_x_kernel,
                    init_move_y_kernel,
                    init_move_x_single_kernel,
                    init_move_y_single_kernel,
                    init_cursor_kernel,
                    init_load_kernel,
                    init_paint_kernel,
                ),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                add_update(update_fluids).in_set(UpdatePhase::Step),
            );
    }
}
