use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_sefirot::display::DisplayPlugin;
use bevy_sefirot::prelude::*;
use nalgebra::Vector2;
use world::fluid::FluidPlugin;

use crate::render::agx::AgXTonemapPlugin;
use crate::render::debug::DebugPlugin;
use crate::render::dither::DitherPlugin;
use crate::render::light::{LightConstants, LightParameters, LightPlugin};
use crate::render::{RenderParameters, RenderPlugin};
use crate::ui::debug::DebugUiPlugin;
use crate::ui::UiPlugin;
use crate::world::physics::{InitData, PhysicsPlugin, NULL_OBJECT};
use crate::world::WorldPlugin;

pub mod prelude;
pub mod render;
pub mod ui;
pub mod utils;
pub mod world;

fn install_eyre() {
    use color_eyre::config::*;
    HookBuilder::blank()
        .capture_span_trace_by_default(true)
        .add_frame_filter(Box::new(|frames| {
            let allowed = &["sefirot", "limbo"];
            frames.retain(|frame| {
                allowed.iter().any(|f| {
                    let name = if let Some(name) = frame.name.as_ref() {
                        name.as_str()
                    } else {
                        return false;
                    };

                    name.starts_with(f)
                })
            });
        }))
        .install()
        .unwrap();
}

fn main() {
    install_eyre();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resizable: false,
                decorations: false,
                resolution: WindowResolution::new(1920.0, 1080.0),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default()))
        .add_plugins(LuisaPlugin {
            device: DeviceType::Cuda,
            ..default()
        })
        .add_plugins(DisplayPlugin::default())
        .add_plugins(WorldPlugin)
        .add_plugins(FluidPlugin)
        .add_plugins(UiPlugin)
        .add_plugins(RenderPlugin::default())
        .add_plugins(AgXTonemapPlugin)
        .add_plugins(DitherPlugin)
        .add_plugins(DebugPlugin)
        .add_plugins(DebugUiPlugin)
        .add_systems(Startup, setup_init_data)
        .insert_resource(Camera {
            position: Vector2::new(128.0, 128.0),
        })
        .add_systems(PreUpdate, (move_camera, update_viewport).chain())
        .run();
}

fn setup_init_data(mut commands: Commands) {
    let mut cells = [[NULL_OBJECT; 256]; 256];
    let platform = 0;
    let block = 1;
    for x in 64..192 {
        for y in 128 - 8..128 + 8 {
            cells[x as usize][y as usize] = platform;
        }
    }
    for x in 0..8 {
        for y in 0..8 {
            cells[x as usize + 66][y as usize + 170] = block;
        }
    }

    // for x in 0..16 {
    //     for y in 0..16 {
    //         cells[x as usize + 66][y as usize + 5] = 2;
    //     }
    // }
    commands.insert_resource(InitData {
        cells,
        object_velocity: vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.7),
        ],
        object_angvel: vec![0.0, 0.0, 0.0],
    });
}

#[derive(Resource)]
struct Camera {
    position: Vector2<f32>,
}

fn move_camera(input: Res<ButtonInput<KeyCode>>, mut camera: ResMut<Camera>) {
    let mut force = Vector2::zeros();
    if input.pressed(KeyCode::KeyA) {
        force.x -= 1.0;
    }
    if input.pressed(KeyCode::KeyD) {
        force.x += 1.0;
    }
    if input.pressed(KeyCode::KeyW) {
        force.y += 1.0;
    }
    if input.pressed(KeyCode::KeyS) {
        force.y -= 1.0;
    }
    camera.position += force;
}

fn update_viewport(
    mut render_parameters: ResMut<RenderParameters>,
    light_constants: Option<Res<LightConstants>>,
    light_parameters: Option<ResMut<LightParameters>>,
    camera: Res<Camera>,
) {
    let position = camera.position;
    render_parameters.view_center = position;
    if let Some(mut lp) = light_parameters {
        lp.set_center(&light_constants.unwrap(), Vector2::repeat(64));
    }
}
