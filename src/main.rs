use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::WindowMode;
use bevy_sefirot::display::DisplayPlugin;
use bevy_sefirot::prelude::*;
use imf::ImfPlugin;
use nalgebra::Vector2;
use physics::{PhysicsPlugin, RigidBodyContext};
use rapier2d::dynamics::{RigidBodyBuilder, RigidBodyHandle};
use rapier2d::geometry::ColliderBuilder;
use render::debug::DebugPlugin;
use render::dither::DitherPlugin;
use render::light::{LightConstants, LightParameters, LightPlugin};
use render::{RenderParameters, RenderPlugin};
use world::WorldPlugin;

mod imf;
mod physics;
mod prelude;
mod render;
pub mod utils;
mod world;

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
                mode: WindowMode::BorderlessFullscreen,
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
        .add_plugins(WorldPlugin)
        .add_plugins(PhysicsPlugin)
        .add_plugins(ImfPlugin)
        .add_plugins(RenderPlugin::default())
        .add_plugins(DitherPlugin)
        .add_plugins(LightPlugin)
        .add_systems(Startup, setup)
        .add_systems(PreUpdate, (apply_player_force, update_viewport).chain())
        .run();
}

#[derive(Component)]
struct Player {
    body: RigidBodyHandle,
}

fn apply_player_force(
    input: Res<ButtonInput<KeyCode>>,
    mut rb_context: ResMut<RigidBodyContext>,
    players: Query<&Player>,
) {
    for player in players.iter() {
        let player = &mut rb_context.bodies[player.body];
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
        if force.norm() > 0.0 {
            let force = force.normalize() * 30.0;
            player.apply_impulse(force, true);
        }
        if input.pressed(KeyCode::Space) {
            player.set_linvel(Vector2::new(player.linvel().x, 20.0), true);
        }
    }
}

#[derive(Component)]
pub struct ActivePlayer;

fn update_viewport(
    mut render_parameters: ResMut<RenderParameters>,
    light_constants: Res<LightConstants>,
    mut light_parameters: ResMut<LightParameters>,
    rb_context: Res<RigidBodyContext>,
    players: Query<&Player, With<ActivePlayer>>,
) {
    let player = players.single();
    let position = rb_context.bodies[player.body].translation();
    render_parameters.view_center = *position;
    light_parameters.set_center(&light_constants, Vector2::repeat(64)); // position.map(|x| x.round() as i32)
}

fn setup(mut commands: Commands, mut rb_context: ResMut<RigidBodyContext>) {
    let body = RigidBodyBuilder::fixed()
        .translation(Vector2::new(64.0, 20.0))
        .build();
    let collider = ColliderBuilder::cuboid(50.0, 6.0).build();
    rb_context.insert2(body, collider);
    let player = RigidBodyBuilder::dynamic()
        .translation(Vector2::new(64.0, 64.0))
        .lock_rotations()
        .build();
    let player_collider = ColliderBuilder::cuboid(5.0, 5.0).build();
    let player = rb_context.insert2(player, player_collider);
    commands.spawn((Player { body: player }, ActivePlayer));
}
