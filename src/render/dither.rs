use nalgebra::DMatrix;

use super::delinearize_pass;
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
    let dim = render_constants.scaling;
    let n = dim.next_power_of_two().ilog2();
    let dim = 1 << n;
    let bayer = bayer(n) / 255.0;
    let texture = device.create_tex2d::<f32>(PixelStorage::Float1, dim, dim, 1);
    // TODO: Make async using copy_from_vec after adding a `RenderInit` phase.
    texture.view(0).copy_from(bayer.as_slice());
    commands.insert_resource(DitherTexture { texture });
}

#[tracked]
fn dither_pass(
    pixel: NonSend<PostprocessData>,
    dither: Res<DitherTexture>,
    render_constants: Res<RenderConstants>,
) {
    let dither = dither
        .texture
        .read(pixel.screen_pos % render_constants.scaling);
    *pixel.color += dither;
}

pub struct DitherPlugin;
impl Plugin for DitherPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_texture)
            .add_systems(BuildPostprocess, dither_pass.after(delinearize_pass));
    }
}
