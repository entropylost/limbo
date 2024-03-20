// Taken from https://www.shadertoy.com/view/cd3XWr
// and https://iolite-engine.com/blog_posts/minimal_agx_implementation

// MIT License
//
// Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// All values used to derive this implementation are sourced from Troy’s initial AgX implementation/OCIO config file available here:
//   https://github.com/sobotka/AgX

use luisa::lang::types::vector::Mat3;

use super::prelude::*;
use crate::prelude::*;

// Mean error^2: 3.6705141e-06
#[tracked]
fn agx_default_contrast_approx(x: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    let x2 = x * x;
    let x4 = x2 * x2;

    15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
        - 0.00232
}

#[tracked]
fn agx(val: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    #[allow(clippy::excessive_precision)]
    let agx_mat = Mat3::from_column_array(&[
        [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
        [0.0784335999999992, 0.878468636469772, 0.0784336],
        [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
    ]);
    let min_ev = -12.47393_f32;
    let max_ev = 4.026069_f32;

    let val = agx_mat.expr() * val;
    let val = val.log2().clamp(min_ev, max_ev);
    let val = (val - min_ev) / (max_ev - min_ev);
    agx_default_contrast_approx(val)
}

#[tracked]
fn agx_eotf(val: Expr<Vec3<f32>>) -> Expr<Vec3<f32>> {
    #[allow(clippy::excessive_precision)]
    let agx_mat_inv = Mat3::from_column_array(&[
        [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
        [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
        [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
    ]);

    agx_mat_inv.expr() * val
    // No need to linearize since outputting to sRGB.
}

#[derive(Debug, Resource, Clone, Copy, PartialEq)]
pub struct AgXConstants {
    pub offset: Vector3<f32>,
    pub slope: Vector3<f32>,
    pub power: Vector3<f32>,
    pub saturation: f32,
}
impl Default for AgXConstants {
    fn default() -> Self {
        Self {
            offset: Vector3::zeros(),
            slope: Vector3::repeat(1.0),
            power: Vector3::repeat(1.0),
            saturation: 1.0,
        }
    }
}
impl AgXConstants {
    pub fn golden() -> Self {
        Self {
            offset: Vector3::zeros(),
            slope: Vector3::new(1.0, 0.9, 0.5),
            power: Vector3::repeat(0.8),
            saturation: 0.8,
        }
    }
    pub fn punchy() -> Self {
        Self {
            offset: Vector3::zeros(),
            slope: Vector3::repeat(1.0),
            power: Vector3::repeat(1.35),
            saturation: 1.4,
        }
    }
}

#[tracked]
fn agx_look(val: Expr<Vec3<f32>>, constants: AgXConstants) -> Expr<Vec3<f32>> {
    let lw = Vec3::new(0.2126, 0.7152, 0.0722);
    let luma = val.dot(lw);

    let offset = Vec3::from(constants.offset);
    let slope = Vec3::from(constants.slope);
    let power = Vec3::from(constants.power);
    let sat = constants.saturation;

    let val = (val * slope + offset).powf(power);
    luma + sat * (val - luma)
}

#[tracked]
fn agx_pass(pixel: NonSend<PostprocessData>, constants: Option<Res<AgXConstants>>) {
    let val = agx(**pixel.color);
    let val = if let Some(constants) = constants {
        agx_look(val, *constants)
    } else {
        val
    };
    *pixel.color = agx_eotf(val);
}

pub struct AgXTonemapPlugin;
impl Plugin for AgXTonemapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(BuildPostprocess, agx_pass.in_set(PostprocessPhase::Tonemap));
    }
}
