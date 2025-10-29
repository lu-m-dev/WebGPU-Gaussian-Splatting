struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) rgba: vec4<f32>,
    @location(1) opacity: f32,
    @location(2) center: vec2<f32>,
    @location(3) radius: vec2<f32>,
    @location(4) conic: vec3<f32>,
};

// struct Splat {
//     //TODO: information defined in preprocess compute shader
// };

import { Splat, CameraUniforms, RenderSettings } from './preprocess.wgsl';

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> splats : array<Splat>;
@group(1) @binding(1)
var<storage, read> sort_indices : array<u32>;
@group(1) @binding(2)
var<storage, read> renderer_settings : RenderSettings;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    let index = sort_indices[instance_index];

    if (index >= renderer_settings.num_points) {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.rgba = vec4<f32>(0.0);
        out.opacity = 0.0;
        out.center = vec2<f32>(0.0);
        out.radius = vec2<f32>(0.0);
        out.conic = vec3<f32>(0.0);
        return out;
    }

    let g = splats[index];

    let center = unpack2x16float(g.center);
    let radius = unpack2x16float(g.size);
    let col_rg = unpack2x16float(g.rgba[0]);
    let col_ba = unpack2x16float(g.rgba[1]);
    let color = vec4<f32>(col_rg.x, col_rg.y, col_ba.x, col_ba.y);

    let conic0 = unpack2x16float(g.conic[0]);
    let conicz = unpack2x16float(g.conic[1]).x;
    let conic = vec3<f32>(conic0.x, conic0.y, conicz);
    let opacity = unpack2x16float(g.opacity).x;

    const normals: array<vec2<f32>, 6> = array<vec2<f32>,6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );

    let offset = normals[vertex_index] * radius;

    out.position = vec4<f32>(center + offset, 0.0f, 1.0f);
    out.rgba = color;
    out.opacity = opacity;
    out.center = center;
    out.radius = radius;
    out.conic = conic;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ndc = (in.position.xy / camera.viewport) * 2.0f - 1.0f;
    let radius = (ndc - in.center) * (camera.viewport / 2.0f);

    let x = in.conic.x;
    let y = in.conic.y;
    let z = in.conic.z;
    let rx = radius.x;
    let ry = radius.y;

    let prod = x * Math.power(rx, 2) + z * Math.power(ry, 2) - 2.0f * y * rx * ry;

    if (prod < 0.0f) {
        return vec4<f32>(0.0f);
    }
    return in.rgba * min(in.opacity * exp(-prod / 2.0f), 0.99f);
}