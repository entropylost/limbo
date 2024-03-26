use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::WindowResolution;
use bevy_sefirot::display::DisplayPlugin;
use bevy_sefirot::prelude::*;
use nalgebra::Vector2;
use rapier2d::dynamics::{RigidBodyBuilder, RigidBodyHandle};
use rapier2d::geometry::ColliderBuilder;
use render::agx::AgXTonemapPlugin;
use ui::debug::DebugUiPlugin;
use ui::UiPlugin;
use world::flow::FlowPlugin;

use crate::render::debug::DebugPlugin;
use crate::render::dither::DitherPlugin;
use crate::render::light::{LightConstants, LightParameters, LightPlugin};
use crate::render::{RenderParameters, RenderPlugin};
use crate::world::impeller::ImpellerPlugin;
use crate::world::physics::{PhysicsPlugin, RigidBodyContext};
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
        .add_plugins(PhysicsPlugin)
        .add_plugins(UiPlugin)
        .add_plugins(ImpellerPlugin)
        .add_plugins(FlowPlugin)
        .add_plugins(RenderPlugin::default())
        .add_plugins(AgXTonemapPlugin)
        .add_plugins(DitherPlugin)
        .add_plugins(LightPlugin)
        .add_plugins(DebugPlugin)
        .add_plugins(DebugUiPlugin)
        .add_plugins(world::tiled_test::TiledTestPlugin)
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
        // player.set_rotation(UnitComplex::from_angle(PI / 2.0), true);
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
            let force = force.normalize() / 2.0;
            player.apply_impulse(force, true);
        }
        if input.pressed(KeyCode::Space) {
            player.set_linvel(Vector2::new(0.0, 0.0), true);
        }
    }
}

#[derive(Component)]
pub struct ActivePlayer;

fn update_viewport(
    mut render_parameters: ResMut<RenderParameters>,
    light_constants: Option<Res<LightConstants>>,
    light_parameters: Option<ResMut<LightParameters>>,
    rb_context: Res<RigidBodyContext>,
    players: Query<&Player, With<ActivePlayer>>,
) {
    let player = players.single();
    let position = rb_context.bodies[player.body].translation();
    render_parameters.view_center = *position;
    if let Some(mut lp) = light_parameters {
        lp.set_center(&light_constants.unwrap(), Vector2::repeat(64));
        // position.map(|x| x.round() as i32)
    }
}

fn setup(mut commands: Commands, mut rb_context: ResMut<RigidBodyContext>) {
    //  let body = RigidBodyBuilder::fixed()
    //      .translation(Vector2::new(20.0, 20.0))
    //      .build();
    //  let collider = ColliderBuilder::cuboid(6.0, 50.0).build();
    //  rb_context.insert2(body, collider);
    let body = RigidBodyBuilder::dynamic()
        .translation(Vector2::new(64.0, 20.0))
        .build();
    let collider = ColliderBuilder::cuboid(50.0, 6.0).build();
    rb_context.insert2(body, collider);

    // 0
    let mut player = RigidBodyBuilder::dynamic()
        .translation(Vector2::new(64.0, 64.0))
        .lock_rotations()
        .build();
    player.activation_mut().linear_threshold = 0.1;
    player.activation_mut().angular_threshold = 0.001;
    let player_collider = ColliderBuilder::cuboid(5.0, 5.0).build();
    let player = rb_context.insert2(player, player_collider);
    commands.spawn((Player { body: player }, ActivePlayer));
}
