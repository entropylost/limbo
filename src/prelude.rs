pub use bevy::prelude::*;
pub use bevy_sefirot::prelude::*;
pub use luisa::lang::types::vector::{Vec2, Vec3, Vec4};
use nalgebra::ComplexField;
pub use nalgebra::{Vector2, Vector3, Vector4};

pub fn sin(x: f32) -> f32 {
    ComplexField::sin(x)
}
pub fn cos(x: f32) -> f32 {
    ComplexField::cos(x)
}
pub fn tan(x: f32) -> f32 {
    ComplexField::tan(x)
}
