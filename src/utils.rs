use std::ops::DerefMut;

use bevy::ecs::schedule::ScheduleLabel;
use bevy_sefirot::MirrorGraph;
use nalgebra::ComplexField;

use crate::prelude::*;

pub fn sin(x: f32) -> f32 {
    ComplexField::sin(x)
}
pub fn cos(x: f32) -> f32 {
    ComplexField::cos(x)
}
pub fn tan(x: f32) -> f32 {
    ComplexField::tan(x)
}

pub fn run_schedule(label: impl ScheduleLabel + Copy) -> impl FnMut(&mut BevyWorld) {
    move |world| {
        world.run_schedule(label);
    }
}

pub fn init<T: Resource + FromWorld>(mut commands: Commands) {
    commands.init_resource::<T>();
}

pub fn execute_graph<T: DerefMut<Target = MirrorGraph> + Resource>(mut graph: ResMut<T>) {
    graph.execute_init();
}

// https://nullprogram.com/blog/2018/07/31/
#[tracked]
pub fn hash(x: Expr<u32>) -> Expr<u32> {
    let x = x.var();
    *x ^= x >> 17;
    *x *= 0xed5ad4bb;
    *x ^= x >> 11;
    *x *= 0xac4c1b51;
    *x ^= x >> 15;
    *x *= 0x31848bab;
    *x ^= x >> 14;
    **x
}

#[tracked]
pub fn rand(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<u32> {
    let input = t + pos.x * 179 + pos.y * 1531 + c * 7919; //* GRID_SIZE * GRID_SIZE * GRID_SIZE;
    hash(input)
}

#[tracked]
pub fn rand_f32(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<f32> {
    rand(pos, t, c).as_f32() / u32::MAX as f32
}
