use bevy::ecs::schedule::ScheduleLabel;
use bevy_sefirot::MirrorGraph;
use sefirot_grid::GridDomain;

use crate::prelude::*;

pub mod direction;
pub mod imf;
pub mod physics;

#[derive(
    ScheduleLabel, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect,
)]
pub struct WorldUpdate;
#[derive(
    ScheduleLabel, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect,
)]
pub struct HostUpdate;

#[derive(
    ScheduleLabel, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect,
)]
pub struct WorldInit;

#[derive(States, Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Reflect)]
pub enum WorldState {
    #[default]
    Running,
    Paused,
}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct InitGraph(pub MirrorGraph);
impl FromWorld for InitGraph {
    fn from_world(world: &mut BevyWorld) -> Self {
        Self(MirrorGraph::from_world(WorldInit, world))
    }
}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct UpdateGraph(pub MirrorGraph);
impl FromWorld for UpdateGraph {
    fn from_world(world: &mut BevyWorld) -> Self {
        Self(MirrorGraph::from_world(WorldUpdate, world))
    }
}

pub fn add_update<
    F: IntoSystem<I, N, M> + 'static,
    I: 'static,
    N: AsNodes + 'static,
    M: 'static,
>(
    f: F,
) -> impl System<In = I, Out = ()> {
    MirrorGraph::add_node::<UpdateGraph, F, I, N, M>(f)
}
pub fn add_init<F: IntoSystem<I, N, M> + 'static, I: 'static, N: AsNodes + 'static, M: 'static>(
    f: F,
) -> impl System<In = I, Out = ()> {
    MirrorGraph::add_node::<InitGraph, F, I, N, M>(f)
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UpdatePhase {
    CopyBodiesFromHost,
    Step,
    CopyBodiesToHost,
}

#[derive(Resource, Deref)]
pub struct World {
    pub domain: GridDomain,
}

impl FromWorld for World {
    fn from_world(_world: &mut BevyWorld) -> Self {
        let domain = GridDomain::new_wrapping([-64, -64], [256, 256]).with_morton();
        World { domain }
    }
}

fn pause_system(
    state: Res<State<WorldState>>,
    mut next: ResMut<NextState<WorldState>>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        next.0 = Some(match **state {
            WorldState::Running => WorldState::Paused,
            WorldState::Paused => WorldState::Running,
        });
    }
}

pub struct WorldPlugin;
impl Plugin for WorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<World>()
            .init_schedule(WorldUpdate)
            .init_schedule(WorldInit)
            .init_schedule(HostUpdate)
            .init_state::<WorldState>()
            .configure_sets(
                WorldUpdate,
                (
                    UpdatePhase::CopyBodiesFromHost,
                    UpdatePhase::Step,
                    UpdatePhase::CopyBodiesToHost,
                )
                    .chain(),
            )
            .add_systems(
                Startup,
                (init_resource::<InitGraph>, init_resource::<UpdateGraph>),
            )
            .add_systems(
                PreUpdate,
                (run_schedule::<WorldInit>, execute_graph::<InitGraph>)
                    .chain()
                    .run_if(run_once()),
            )
            .add_systems(
                Update,
                (
                    (
                        run_schedule::<WorldUpdate>,
                        (execute_graph::<UpdateGraph>, run_schedule::<HostUpdate>),
                    )
                        .chain()
                        .run_if(in_state(WorldState::Running)),
                    pause_system,
                )
                    .chain(),
            );
    }
}
