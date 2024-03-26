use std::sync::Arc;

use sefirot::mapping::buffer::StaticDomain;
use sefirot_grid::offset::OffsetDomain;
use sefirot_grid::tiled::{TileArray, TileArrayParameters, TileDomain};

use crate::prelude::*;

#[derive(Resource)]
pub struct TiledTestFields {
    pub domain: OffsetDomain<TileDomain>,
    tiles: Arc<TileArray>,
    pub data_field: AField<bool, Cell>,
    _fields: FieldSet,
}

#[kernel]
fn startup_kernel(device: Res<Device>, fields: Res<TiledTestFields>) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<0>::new(), &|el| {
        let cell = el.at(Vec2::splat_expr(64_i32));
        *fields.data_field.var(&cell) = true;
        fields.domain.activate(&cell);
    })
}

#[kernel]
fn fill_kernel(
    device: Res<Device>,
    world: Res<World>,
    fields: Res<TiledTestFields>,
) -> Kernel<fn()> {
    Kernel::build(&device, &fields.domain, &|cell| {
        if !fields.data_field.expr(&cell) {
            return;
        }
        for dir in GridDirection::iter_all() {
            let neighbor = world.in_dir(&cell, dir);
            if world.contains(&neighbor) {
                if !fields.data_field.expr(&neighbor) {
                    *fields.data_field.var(&neighbor) = true;
                    fields.domain.activate(&neighbor);
                }
            }
        }
        fields.domain.activate(&cell);
    })
}

fn setup_fields(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let tiles = TileArray::new(TileArrayParameters {
        device: device.clone(),
        tile_size: 8,
        array_size: [32, 32],
        max_active_tiles: 32 * 32,
    });
    let data_field = fields.create_bind("tiled-test-data", world.create_buffer(&device));
    let domain = world.offset(tiles.allocate());

    commands.insert_resource(TiledTestFields {
        domain,
        tiles,
        data_field,
        _fields: fields,
    });
}

fn update_tiled(mut t: Local<u32>, fields: Res<TiledTestFields>) -> impl AsNodes {
    *t += 1;
    if *t == 1 {
        Some((startup_kernel.dispatch(), fields.tiles.update()).chain())
    } else if *t % 16 == 0 {
        Some((fill_kernel.dispatch(), fields.tiles.update()).chain())
    } else {
        None
    }
}

pub struct TiledTestPlugin;
impl Plugin for TiledTestPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_fields)
            .add_systems(InitKernel, (init_startup_kernel, init_fill_kernel))
            .add_systems(WorldUpdate, add_update(update_tiled));
    }
}
