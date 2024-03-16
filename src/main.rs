use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_sefirot::display::DisplayPlugin;
use bevy_sefirot::prelude::*;
use render::RenderPlugin;

mod prelude;
mod render;

fn main() {
    color_eyre::install().unwrap();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resizable: false,
                resolution: WindowResolution::new(512.0 * 2.0, 512.0 * 2.0),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default()))
        .add_plugins(LuisaPlugin {
            device: "cuda".to_string(),
            ..default()
        })
        .add_plugins(DisplayPlugin)
        .add_plugins(RenderPlugin::default())
        .run();
}
