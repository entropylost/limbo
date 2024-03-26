use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{RenderGraph, RenderLabel};
use bevy::render::render_resource::{
    LoadOp, Operations, RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
};
use bevy::render::view::ExtractedWindows;
use bevy::render::RenderApp;
use bevy::window::{PresentMode, WindowResolution};
use bevy_egui::render_systems::EguiPass;
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

    commands.insert_resource(UiWindowId(ui_window_id));
}

fn add_ui_node(window: Option<Res<UiWindowId>>, mut graph: ResMut<RenderGraph>) {
    let Some(window) = window else {
        return;
    };
    if !window.is_added() {
        return;
    }
    graph.add_node(ClearLabel, ClearNode);
    graph.add_node_edge(CameraDriverLabel, ClearLabel);
    graph.add_node_edge(
        ClearLabel,
        EguiPass {
            window_index: window.0.index(),
            window_generation: window.0.generation(),
        },
    );
}

pub struct UiPlugin;
impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::NONE))
            .add_plugins(ExtractResourcePlugin::<UiWindowId>::default())
            .add_plugins(EguiPlugin)
            .add_systems(Startup, create_window_system);
        app.sub_app_mut(RenderApp)
            .add_systems(bevy::render::Render, add_ui_node);
        // TODO: Make a Ui Schedule / systemset or something.
    }
}

#[derive(Resource, Debug, Hash, PartialEq, Eq, Clone, Copy, ExtractResource)]
struct UiWindowId(Entity);

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, RenderLabel)]
struct ClearLabel;

struct ClearNode;
impl bevy::render::render_graph::Node for ClearNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &BevyWorld,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let Some(UiWindowId(ui_window_id)) = world.get_resource::<UiWindowId>() else {
            return Ok(());
        };
        let Some(window) = world
            .resource::<ExtractedWindows>()
            .windows
            .get(ui_window_id)
        else {
            return Ok(());
        };

        let swap_chain_texture_view = window.swap_chain_texture_view.as_ref().unwrap();

        render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("clear render pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: swap_chain_texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::NONE.into()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        Ok(())
    }
}
