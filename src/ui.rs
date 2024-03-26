use bevy::render::camera::RenderTarget;
use bevy::window::{PresentMode, WindowRef, WindowResolution};
use bevy_egui::{EguiContext, EguiPlugin};

use crate::prelude::*;

pub mod debug;

pub type UiContext<'w, 's, 'a> = Query<'w, 's, &'a mut EguiContext, With<UiWindow>>;

#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct UiWindow;

fn create_window_system(mut commands: Commands) {
    let ui_window_id = commands
        .spawn(Window {
            title: "Ui Window".to_string(),
            transparent: true,
            decorations: false,
            resizable: false,
            resolution: WindowResolution::new(1920.0, 1080.0),
            present_mode: PresentMode::AutoNoVsync,
            ..default()
        })
        .insert(UiWindow)
        .id();

    commands.spawn(Camera3dBundle {
        camera: Camera {
            target: RenderTarget::Window(WindowRef::Entity(ui_window_id)),
            ..Default::default()
        },
        ..Default::default()
    });
}

pub struct UiPlugin;
impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::NONE))
            .add_plugins(EguiPlugin)
            .add_systems(Startup, create_window_system);
        // TODO: Make a Ui Schedule / systemset or something.
    }
}
