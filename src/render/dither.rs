use nalgebra::DMatrix;

use super::delinearize;
use super::prelude::*;
use crate::prelude::*;

fn bayer_iter(input: DMatrix<u32>) -> DMatrix<u32> {
    let n = input.nrows();

    let mut output = DMatrix::<u32>::zeros(n * 2, n * 2);
    for i in 0..n {
        for j in 0..n {
            let v = input[(i, j)] * 4;
            output[(i, j)] = v;
            output[(i + n, j)] = v + 2;
            output[(i, j + n)] = v + 3;
            output[(i + n, j + n)] = v + 1;
        }
    }
    output
}

fn bayer(n: u32) -> DMatrix<f32> {
    let mut output = DMatrix::<u32>::zeros(1, 1);
    for _ in 0..n {
        output = bayer_iter(output);
    }
    output.map(|x| x as f32 / (1 << (2 * n)) as f32 - 0.5)
}

#[derive(Resource)]
struct DitherTexture {
    texture: Tex2d<f32>,
}

fn setup_texture(
    mut commands: Commands,
    device: Res<Device>,
    render_constants: Res<RenderConstants>,
) {
    let dim = render_constants.upscale_factor;
    let n = dim.ilog2();
    let bayer = bayer(n) / 255.0;
    let texture = device.create_tex2d::<f32>(PixelStorage::Float1, dim, dim, 1);
    // TODO: Make async using copy_from_vec after adding a `RenderInit` phase.
    texture.view(0).copy_from(bayer.as_slice());
    commands.insert_resource(DitherTexture { texture });
}

#[kernel]
fn dither_kernel(
    device: Res<Device>,
    render: Res<RenderFields>,
    render_constants: Res<RenderConstants>,
    dither: Res<DitherTexture>,
) -> Kernel<fn()> {
    Kernel::build(&device, &render.screen_domain, &|el| {
        let dither = dither.texture.read(*el % render_constants.upscale_factor);
        *render.screen_color.var(&el) += dither;
    })
}
fn dither() -> impl AsNodes {
    dither_kernel.dispatch()
}

pub struct DitherPlugin;
impl Plugin for DitherPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_texture)
            .add_systems(InitKernel, init_dither_kernel)
            .add_systems(
                Render,
                add_render(dither)
                    .in_set(RenderPhase::Postprocess)
                    .after(delinearize),
            );
    }
}
