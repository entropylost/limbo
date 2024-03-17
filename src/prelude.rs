pub use bevy::ecs::world::World as BevyWorld;
pub use bevy::prelude::*;
pub use bevy_sefirot::prelude::*;
pub use bevy_sefirot::AsNodesStatic as AsNodes;
pub use luisa::lang::types::vector::{Vec2, Vec3, Vec4};
use nalgebra::ComplexField;
pub use nalgebra::Vector2;
pub use sefirot::graph::AsNodes as AsNodesExt;

pub use crate::world::{
    add_init, add_update, HostUpdate, UpdatePhase, World, WorldInit, WorldUpdate,
};

pub fn sin(x: f32) -> f32 {
    ComplexField::sin(x)
}
pub fn cos(x: f32) -> f32 {
    ComplexField::cos(x)
}
pub fn tan(x: f32) -> f32 {
    ComplexField::tan(x)
}
