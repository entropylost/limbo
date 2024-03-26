use sefirot::field::FieldId;
use sefirot::track_nc;

use super::UiContext;
use crate::prelude::*;
use crate::render::debug::DebugParameters;
use crate::render::light::LightParameters;
use crate::world::flow::FlowFields;
use crate::world::impeller::ImpellerFields;
use crate::world::physics::{PhysicsFields, NULL_OBJECT};

#[derive(Resource, Debug)]
pub struct DebugUiState {
    activate_debug_render: bool,
    current_index: usize,
    debug_fields: Vec<(String, FieldId)>,
    _fields: FieldSet,
}
impl FromWorld for DebugUiState {
    fn from_world(world: &mut BevyWorld) -> Self {
        let mut fields = FieldSet::new();
        let mut debug_fields = vec![];
        if let Some(physics) = world.get_resource::<PhysicsFields>() {
            let object: EField<u32, Cell> = *physics.object;
            let debug_object: EField<Vec3<f32>, Cell> = fields.create_bind(
                "debug-object",
                object.map(track_nc!(|x| {
                    if x == NULL_OBJECT {
                        Vec3::splat_expr(0.0_f32)
                    } else {
                        let x = x.cast_f32();

                        Vec3::expr(x.cos(), x.sin(), (x * 0.1).sin() + 0.5).normalize()
                    }
                })),
            );
            debug_fields.push(("Object", debug_object.id()))
        }
        if let Some(imf) = world.get_resource::<ImpellerFields>() {
            let mass: EField<f32, Cell> = *imf.mass;
            let debug_mass: EField<Vec3<f32>, Cell> = fields.create_bind(
                "debug-mass",
                mass.map(track_nc!(|x| { Vec3::splat_expr(x) })),
            );
            debug_fields.push(("Mass", debug_mass.id()));

            let velocity: EField<Vec2<f32>, Cell> = *imf.velocity;
            let debug_velocity: EField<Vec3<f32>, Cell> = fields.create_bind(
                "debug-velocity",
                velocity.map(track_nc!(|v| { Vec3::expr(v.x + 0.5, v.y + 0.5, 0.0) })),
            );
            debug_fields.push(("Velocity", debug_velocity.id()));
        }
        if let Some(flow) = world.get_resource::<FlowFields>() {
            let activation: EField<bool, Cell> = *flow.activation;
            debug_fields.push(("Flow Activation", activation.id()));
        }
        Self {
            activate_debug_render: false,
            current_index: 0,
            debug_fields: debug_fields
                .into_iter()
                .map(|(name, field)| (name.to_string(), field))
                .collect(),
            _fields: fields,
        }
    }
}

fn activate_renders(
    state: Res<DebugUiState>,
    mut debug_params: ResMut<DebugParameters>,
    light_params: Option<ResMut<LightParameters>>,
) {
    if let Some(mut light_params) = light_params {
        light_params.running = !state.activate_debug_render;
        debug_params.running = state.activate_debug_render;
    }
    debug_params.active_field = state.debug_fields[state.current_index].1;
}

fn render_ui(mut state: ResMut<DebugUiState>, mut ctx: UiContext) {
    let DebugUiState {
        activate_debug_render,
        debug_fields,
        current_index,
        ..
    } = &mut *state;
    egui::Window::new("Debug Render").show(ctx.single_mut().get_mut(), |ui| {
        if ui.button("Activate Debug Render").clicked() {
            *activate_debug_render = !*activate_debug_render;
        }
        for (i, (name, _)) in debug_fields.iter().enumerate() {
            ui.radio_value(current_index, i, name);
        }
    });
}

pub struct DebugUiPlugin;
impl Plugin for DebugUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostStartup, init_resource::<DebugUiState>)
            .add_systems(PostUpdate, (render_ui, activate_renders).chain());
    }
}