use std::ops::DerefMut;

use bevy::ecs::schedule::ScheduleLabel;
use bevy_sefirot::MirrorGraph;
use nalgebra::ComplexField;
use sefirot::tracked_nc;

use crate::prelude::*;

#[cfg(feature = "timed")]
static TIMINGS: once_cell::sync::Lazy<parking_lot::Mutex<std::collections::BTreeMap<String, f32>>> =
    once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(std::collections::BTreeMap::new()));
#[cfg(feature = "timed")]
static TIME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn sin(x: f32) -> f32 {
    ComplexField::sin(x)
}
pub fn cos(x: f32) -> f32 {
    ComplexField::cos(x)
}
pub fn tan(x: f32) -> f32 {
    ComplexField::tan(x)
}

pub fn run_schedule<L: ScheduleLabel + Default>(world: &mut BevyWorld) {
    world.run_schedule(L::default());
}

pub fn init_resource<T: Resource + FromWorld>(mut commands: Commands) {
    commands.init_resource::<T>();
}

pub fn execute_graph<T: DerefMut<Target = MirrorGraph> + Resource>(mut graph: ResMut<T>) {
    #[cfg(feature = "trace")]
    graph.execute_trace();
    #[cfg(all(feature = "debug", not(feature = "trace")))]
    graph.execute_dbg();
    #[cfg(all(not(feature = "trace"), not(feature = "debug"), not(feature = "timed")))]
    graph.execute();
    #[cfg(feature = "timed")]
    {
        TIME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut timings = TIMINGS.lock();
        let these_timings = graph.execute_timed();
        for (name, time) in these_timings.iter() {
            let entry = timings.entry(name.clone()).or_insert(0.0);
            *entry = *entry * 0.99 + *time * 0.01;
        }

        if TIME.load(std::sync::atomic::Ordering::Relaxed) % 1000 == 0 {
            for (name, time) in timings.iter() {
                println!("{}: {}", name, time);
            }
        }
    }
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
    let input = pos.x + pos.y * 256 + c * 7919 + t * 2796203; //* GRID_SIZE * GRID_SIZE * GRID_SIZE;
    hash(input)
}

#[tracked]
pub fn rand_f32(pos: Expr<Vec2<u32>>, t: Expr<u32>, c: u32) -> Expr<f32> {
    rand(pos, t, c).as_f32() / u32::MAX as f32
}

/*
Add this one as well.
// https://github.com/markjarzynski/pcg3d
uint3 pcg3d(uint3 v) {
    v = v * 1664525u + 1013904223u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v ^= v>>16u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    return v;
}
*/

pub trait Cross<T> {
    type Output;
    fn cross(&self, other: T) -> Self::Output;
}
impl Cross<Expr<Vec2<f32>>> for Expr<Vec2<f32>> {
    type Output = Expr<f32>;
    #[tracked_nc]
    fn cross(&self, other: Expr<Vec2<f32>>) -> Self::Output {
        self.x * other.y - self.y * other.x
    }
}
impl Cross<Expr<f32>> for Expr<Vec2<f32>> {
    type Output = Expr<Vec2<f32>>;
    #[tracked_nc]
    fn cross(&self, other: Expr<f32>) -> Self::Output {
        Vec2::expr(self.y * other, -self.x * other)
    }
}
impl Cross<Expr<Vec2<f32>>> for Expr<f32> {
    type Output = Expr<Vec2<f32>>;
    #[tracked_nc]
    fn cross(&self, other: Expr<Vec2<f32>>) -> Self::Output {
        Vec2::expr(-*self * other.y, self * other.x)
    }
}
