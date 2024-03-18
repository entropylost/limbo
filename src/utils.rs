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
