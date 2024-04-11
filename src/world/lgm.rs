use luisa::lang::functions::sync_block;
use luisa::lang::types::shared::Shared;
use sefirot::mapping::buffer::StaticDomain;

use crate::prelude::*;

#[derive(Resource)]
pub struct LgmFields {
    domain: StaticDomain<3>,
    pub dirs: VEField<Vec4<u32>, Vec3<u32>>,
    pub next_dirs: VEField<Vec4<u32>, Vec3<u32>>,
    pub walls: VEField<Vec2<u32>, Vec3<u32>>,
    pub rendered: VField<bool, Cell>,
    _fields: FieldSet,
}

fn setup_lgm(mut commands: Commands, device: Res<Device>, world: Res<World>) {
    let mut fields = FieldSet::new();
    let domain = StaticDomain::<3>::new(32, 32, 64);
    let lgm = LgmFields {
        domain,
        dirs: fields.create_bind("lgm-dirs", domain.create_tex3d(&device)),
        next_dirs: fields.create_bind("lgm-next-dirs", domain.create_tex3d(&device)),
        walls: fields.create_bind("lgm-walls", domain.create_tex3d(&device)),
        rendered: *fields.create_bind("lgm-rendered", world.create_buffer(&device)),
        _fields: fields,
    };
    commands.insert_resource(lgm);
}

#[kernel(run)]
fn update_kernel(device: Res<Device>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(
        &device,
        &StaticDomain::<3>::new(lgm.domain.width() - 2, lgm.domain.height() - 2, 64),
        &|ix| {
            set_block_size([1, 1, 64]);
            let ix = ix.at(*ix + Vec3::expr(1, 1, 0));

            let swap = Shared::<u64>::new(64);

            let transpose = |mut x: Expr<u64>| -> Expr<u64> {
                let masks: [u64; 6] = [
                    0x5555555555555555,
                    0x3333333333333333,
                    0x0F0F0F0F0F0F0F0F,
                    0x00FF00FF00FF00FF,
                    0x0000FFFF0000FFFF,
                    0x00000000FFFFFFFF,
                ];
                let ix = ix.z.cast_u64();
                for (i, mask) in masks.into_iter().enumerate() {
                    let pow = 1_u64 << i;
                    let lvl = ix & pow; // 0 if the flipped block is on the "right".
                    swap.write(ix, (x >> (lvl ^ pow)) & mask);
                    sync_block();
                    let y = swap.read(ix ^ pow);
                    x = (x & (mask << lvl)) | (y << (lvl ^ pow));
                }
                x
            };

            // Note that if ix is between 16 and 48, the middle part of the dirs is shared.
            let offset = (16 + ix.z) % 32;
            let xo = (ix.y * 32 - 16 + ix.z) / 32;
            let dir02 = (lgm
                .dirs
                .expr(&ix.at(Vec3::expr(ix.x - 1, xo, offset)))
                .xz()
                .cast_u64()
                >> 16)
                | (lgm
                    .dirs
                    .expr(&ix.at(Vec3::expr(ix.x, xo, offset)))
                    .xz()
                    .cast_u64()
                    << 16)
                | (lgm
                    .dirs
                    .expr(&ix.at(Vec3::expr(ix.x + 1, xo, offset)))
                    .xz()
                    .cast_u64()
                    << 48);
            let yo = (ix.x * 32 - 16 + ix.z) / 32;
            let dir13 = (lgm
                .dirs
                .expr(&ix.at(Vec3::expr(yo, ix.y - 1, offset)))
                .yw()
                .cast_u64()
                >> 16)
                | (lgm
                    .dirs
                    .expr(&ix.at(Vec3::expr(yo, ix.y, offset)))
                    .yw()
                    .cast_u64()
                    << 16)
                | (lgm
                    .dirs
                    .expr(&ix.at(Vec3::expr(yo, ix.y + 1, offset)))
                    .yw()
                    .cast_u64()
                    << 48);

            let dir0 = dir02.x.var();
            let dir1 = dir13.x.var();
            let dir2 = dir02.y.var();
            let dir3 = dir13.y.var();

            let wall0 = 0_u64.expr();
            (lgm.walls
                .expr(&ix.at(Vec3::expr(ix.x - 1, xo, offset)))
                .x
                .cast_u64()
                >> 16)
                | (lgm
                    .walls
                    .expr(&ix.at(Vec3::expr(ix.x, xo, offset)))
                    .x
                    .cast_u64()
                    << 16)
                | (lgm
                    .walls
                    .expr(&ix.at(Vec3::expr(ix.x + 1, xo, offset)))
                    .x
                    .cast_u64()
                    << 48);
            let wall1 = 0_u64.expr();
            (lgm.walls
                .expr(&ix.at(Vec3::expr(yo, ix.y - 1, offset)))
                .y
                .cast_u64()
                >> 16)
                | (lgm
                    .walls
                    .expr(&ix.at(Vec3::expr(yo, ix.y, offset)))
                    .y
                    .cast_u64()
                    << 16)
                | (lgm
                    .walls
                    .expr(&ix.at(Vec3::expr(yo, ix.y + 1, offset)))
                    .y
                    .cast_u64()
                    << 48);

            for _i in 0..1 {
                *dir0 <<= 1;
                *dir1 <<= 1;
                *dir2 >>= 1;
                *dir3 >>= 1;
                let xf = (dir0 ^ dir2) & wall0;
                let yf = (dir1 ^ dir3) & wall1;
                *dir0 ^= xf;
                *dir1 ^= yf;
                *dir2 ^= xf;
                *dir3 ^= yf;
                let xe = !(dir0 | dir2);
                let ye = !(dir1 | dir3);
                let xe2 = transpose(xe);
                let ye2 = transpose(ye);

                let xo = dir0 & dir2 & ye2;
                let yo = dir1 & dir3 & xe2;
                *dir0 ^= xo;
                *dir1 ^= yo;
                *dir2 ^= xo;
                *dir3 ^= yo;
                let xo2 = transpose(xo);
                let yo2 = transpose(yo);

                *dir0 ^= yo2;
                *dir1 ^= xo2;
                *dir2 ^= yo2;
                *dir3 ^= xo2;
            }

            if ix.z >= 16 && ix.z < 48 {
                *lgm.next_dirs.var(&ix.at(Vec3::expr(ix.x, ix.y, ix.z - 16))) =
                    (Vec4::expr(dir0, dir1, dir2, dir3) >> 16).cast_u32();
            }
        },
    )
}

#[kernel(run)]
fn load_wall_kernel(device: Res<Device>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(&device, &lgm.domain, &|ix| {
        if (ix.xy() <= 1).any()
            || (ix.xy() >= Vec2::new(lgm.domain.width() - 2, lgm.domain.height() - 2)).any()
        {
            *lgm.walls.var(&ix) = Vec2::splat_expr(u32::MAX);
        }
    })
}

#[kernel(run)]
fn load_kernel(device: Res<Device>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(&device, &StaticDomain::<1>::new(32), &|ix| {
        *lgm.dirs.var(&ix.at(Vec3::expr(3, 3, *ix))) = Vec4::splat_expr(u32::MAX);
        *lgm.dirs.var(&ix.at(Vec3::expr(3, 4, *ix))) = Vec4::splat_expr(u32::MAX);
        *lgm.dirs.var(&ix.at(Vec3::expr(4, 4, *ix))) = Vec4::splat_expr(u32::MAX);
        *lgm.dirs.var(&ix.at(Vec3::expr(5, 5, *ix))) = Vec4::splat_expr(u32::MAX);
    })
}

// Can replace with actual copy.
#[kernel(run)]
fn copy_kernel(device: Res<Device>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(&device, &lgm.domain, &|ix| {
        *lgm.dirs.var(&ix) = lgm.next_dirs.expr(&ix);
    })
}

#[kernel(run)]
fn render_kernel(device: Res<Device>, world: Res<World>, lgm: Res<LgmFields>) -> Kernel<fn()> {
    Kernel::build(&device, &**world, &|cell| {
        let block = (cell.cast_u32() / 32) % Vec2::new(lgm.domain.width(), lgm.domain.height());
        let pos = cell.cast_u32() % 32;
        let cx = cell.at(Vec3::expr(block.x, block.y, pos.x));
        let cy = cell.at(Vec3::expr(block.x, block.y, pos.y));
        let px = pos.x;
        let py = pos.y;
        *lgm.rendered.var(&cell) = (lgm.walls.expr(&cy).x & (1 << px) != 0)
            || (lgm.dirs.expr(&cy).xz() & (1 << px) != 0).any()
            || (lgm.dirs.expr(&cx).yw() & (1 << py) != 0).any();
    })
}

pub struct LgmPlugin;
impl Plugin for LgmPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_lgm)
            .add_systems(
                InitKernel,
                (
                    init_update_kernel,
                    init_load_kernel,
                    init_render_kernel,
                    init_copy_kernel,
                    init_load_wall_kernel,
                ),
            )
            .add_systems(WorldInit, (add_init(load), add_init(load_wall)))
            .add_systems(
                WorldUpdate,
                (add_update(update), add_update(copy), add_update(render))
                    .chain()
                    .in_set(UpdatePhase::Step),
            );
    }
}
