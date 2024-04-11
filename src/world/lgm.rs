use std::array::from_fn;

use luisa::lang::functions::sync_block;
use luisa::lang::types::shared::Shared;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;

#[derive(Resource)]
pub struct LgmFields {
    domain: StaticDomain<1>,
    pub dirs: [VEField<u64, u32>; 4],
    pub walls: [VEField<u64, u32>; 2],
    pub rendered: VField<bool, Cell>,
    _fields: FieldSet,
}

fn setup_lgm(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let domain = StaticDomain::<1>::new(64);
    let dirs = from_fn(|i| {
        fields.create_bind(
            format!("lgm-dir-{}", i),
            domain.create_buffer::<u64>(&device),
        )
    });
    let mut wall_data = [1 | (1 << 63); 64];
    wall_data[0] = u64::MAX;
    wall_data[63] = u64::MAX;
    let wall = from_fn(|i| {
        fields.create_bind(
            format!("lgm-wall-{}", i),
            domain.map_buffer::<u64>(device.create_buffer_from_slice(&wall_data)),
        )
    });
    let lgm = LgmFields {
        domain,
        dirs,
        walls: wall,
        rendered: *fields.create_bind("lgm-rendered", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(lgm);
}

#[kernel(run)]
fn update_kernel(device: Res<Device>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(&device, &lgm.domain, &|ix| {
        set_block_size([64, 1, 1]);
        let byte_rev = from_fn::<_, 256, _>(|i| (i as u8).reverse_bits() as u32).expr();
        let swap = Shared::<u64>::new(64);
        let dirs = lgm.dirs.map(|dir| dir.var(&ix));
        let walls = lgm.walls.map(|wall| wall.expr(&ix));
        *dirs[0] <<= 1;
        *dirs[1] <<= 1;
        *dirs[2] >>= 1;
        *dirs[3] >>= 1;
        let xf = (dirs[0] ^ dirs[2]) & walls[0];
        let yf = (dirs[1] ^ dirs[3]) & walls[1];
        *dirs[0] ^= xf;
        *dirs[1] ^= yf;
        *dirs[2] ^= xf;
        *dirs[3] ^= yf;
        let xo = dirs[0] & dirs[2];
        let yo = dirs[1] & dirs[3];
        *dirs[0] ^= xo;
        *dirs[1] ^= yo;
        *dirs[2] ^= xo;
        *dirs[3] ^= yo;
        // Begin transpose
        swap.write(*ix, xo);
        sync_block();
        let xoi = swap.read(63 - *ix);
        let mut xo2 = 0u64.expr();
        let r = 0..8u64;
        for i in r {
            xo2 = xo2 | (byte_rev.read((xoi >> (i * 8)) & 0xFF).cast_u64() << (i * 8));
        }

        swap.write(*ix, yo);
        sync_block();
        let yoi = swap.read(63 - *ix);
        let mut yo2 = 0u64.expr();
        let r = 0..8u64;
        for i in r {
            yo2 = yo2 | (byte_rev.read((yoi >> (i * 8)) & 0xFF).cast_u64() << (i * 8));
        }

        *dirs[0] ^= yo2;
        *dirs[1] ^= xo2;
        *dirs[2] ^= yo2;
        *dirs[3] ^= xo2;
    })
}

#[kernel(run)]
fn load_kernel(device: Res<Device>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(32), &|ix| {
        let x = *ix + 16;
        let r = 0..4;
        for i in r {
            *lgm.dirs[i].var(&ix.at(x)) = ((1_u64 << 32) - 1_u64) << 16;
        }
    })
}

#[kernel(run)]
fn render_kernel(device: Res<Device>, world: Res<World>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let pos = cell.cast_u32() % 64;
        *lgm.rendered.var(&cell) = (lgm.walls[0].expr(&cell.at(pos.x)) & (1 << pos.y.cast_u64())
            != 0)
            || (lgm.dirs[0].expr(&cell.at(pos.x)) & (1 << pos.y.cast_u64()) != 0)
            || (lgm.dirs[1].expr(&cell.at(pos.y)) & (1 << pos.x.cast_u64()) != 0)
            || (lgm.dirs[2].expr(&cell.at(pos.x)) & (1 << pos.y.cast_u64()) != 0)
            || (lgm.dirs[3].expr(&cell.at(pos.y)) & (1 << pos.x.cast_u64()) != 0);
    })
}

pub struct LgmPlugin;
impl Plugin for LgmPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_lgm)
            .add_systems(
                InitKernel,
                (init_update_kernel, init_load_kernel, init_render_kernel),
            )
            .add_systems(WorldInit, add_init(load))
            .add_systems(
                WorldUpdate,
                (add_update(update), add_update(render))
                    .chain()
                    .in_set(UpdatePhase::Step),
            );
    }
}
