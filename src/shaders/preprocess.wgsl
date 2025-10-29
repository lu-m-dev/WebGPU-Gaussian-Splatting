const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
    num_points: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    center: u32,
    size: u32,
    rgba: array<u32,2>,
    opacity: u32,
    conic: array<u32,2>,
};

//TODO: bind your data here
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(2)
var<storage, read> sh_buf : array<u32>;

@group(1) @binding(0)
var<storage, read_write> splats : array<Splat>;
@group(1) @binding(1)
var<uniform> settings: RenderSettings;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let mod = c_idx % 2u;
    let ind = splat_idx * 24u + (c_idx / 2) * 3u + mod;
    let rg = unpack2x16float(sh_buf[ind]);
    let ba = unpack2x16float(sh_buf[ind + 1u]);
    if (mod == 0u) {
        return vec3f(rg.x, rg.y, ba.x);
    }
    return vec3f(rg.y, ba.x, ba.y);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u32(settings.num_points)) {
        return;
    }

    let g = gaussians[idx];
    let xy = unpack2x16float(g.pos_opacity[0]);
    let zw = unpack2x16float(g.pos_opacity[1]);
    let pos_world = vec4<f32>(xy.x, xy.y, zw.x, 1.0);

    let view = camera.view * pos_world;
    let clip = camera.proj * view;
    let ndc = clip.xy / clip.w;

    if (abs(ndc.x) > 1.2 || abs(ndc.y) > 1.2 || view.z < 0.0f) {
        return;
    }
   
    let r01 = unpack2x16float(g.rot[0]);
    let r23 = unpack2x16float(g.rot[1]);
    let rot_q = vec4<f32>(r01.y, r23.x, r23.y, r01.x);

    let s01 = unpack2x16float(g.scale[0]);
    let s23 = unpack2x16float(g.scale[1]);
    let scale_log = vec3<f32>(s01.x, s01.y, s23.x);
    let scale_actual = exp(scale_log);

    let x = rot_q.x;
    let y = rot_q.y;
    let z = rot_q.z;
    let w = rot_q.w;
    let mat_cov3d = mat3x3<f32>(
        1.0f - 2.0f * (y*y + z*z), 2.0f * (x*y - w*z), 2.0f * (x*z + w*y),
        2.0f * (x*y + w*z), 1.0f - 2.0f * (x*x + z*z), 2.0f * (y*z - w*x),
        2.0f * (x*z - w*y), 2.0f * (y*z + w*x), 1.0f - 2.0f * (x*x + y*y),
    );

    let gaussian_mult = settings.gaussian_scaling;
    let mat_scale = mat3x3<f32>(
        scale_actual.x * gaussian_mult, 0.0f, 0.0f,
        0.0f, scale_actual.y * gaussian_mult, 0.0f,
        0.0f, 0.0f, scale_actual.z * gaussian_mult,
    );

    let mat_sigma = transpose(mat_cov3d) * power(mat_scale, 2.0f) * mat_cov3d;
    let mat_view = mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz);
    let mat_view_inv = transpose(mat_view);

    let x_cam = view.x;
    let y_cam = view.y;
    let z_cam = view.z;
    let fx = camera.focal.x;
    let fy = camera.focal.y;
    let mat_jacob = mat3x3<f32>(
        fx / z_cam, 0.0f, -fx * x_cam / (z_cam * z_cam),
        0.0f, fy / z_cam, -fy * y_cam / (z_cam * z_cam),
        0.0f, 0.0f, 0.0f,
    );

    let mat_trans = mat_view_inv * mat_jacob;
    var mat_cov2d = transpose(mat_trans) * mat_sigma * mat_trans;

    let e00 = mat_cov2d[0][0];
    let e01 = mat_cov2d[0][1];
    let e11 = mat_cov2d[1][1];

    let det = e00 * e11 - Math.power(e01, 2.0f);
    let val = (e00 + e11) / 2.0f;
    let radius = Math.ceil(3.0f * sqrt(val + sqrt(max(0.1f, Math.power(val, 2.0f) - det))));

    let rx_ndc = radius * (2.0f / camera.viewport.x);
    let ry_ndc = radius * (2.0f / camera.viewport.y);

    let packed_center = pack2x16float(ndc);
    let packed_radius = pack2x16float(vec2<f32>(rx_ndc, ry_ndc));

    let write_idx = atomicAdd(&sort_infos.keys_size, 1u);

    let view_dir = normalize(pos_world.xyz - camera.view_inv[3].xyz);
    let color = computeColorFromSH(view_dir, idx, u32(settings.sh_deg));
    let packed_rg = pack2x16float(color.xy);
    let packed_ba = pack2x16float(vec2<f32>(color.z, 1.0));

    let det_inv = 1.0f / det;
    let conic = vec3<f32>(e11 * det_inv, -e01 * det_inv, e00 * det_inv);
    let packed_conic_xy = pack2x16float(conic.xy);
    let packed_conic_z_pad = pack2x16float(vec2<f32>(conic.z, 0.0f));

    let opacity_sigmoid = 1.0f / (1.0f + exp(-b.y));
    let packed_opacity_pad = pack2x16float(vec2<f32>(opacity_sigmoid, 0.0f));
    let depth_bits = bitcast<u32>(-depthDetect);
    let sort_key = 0xFFFFFFFFu - depth_bits;

    splats[write_idx].center = packed_center;
    splats[write_idx].size = packed_radius;
    splats[write_idx].rgba[0] = packed_rg;
    splats[write_idx].rgba[1] = packed_ba;
    splats[write_idx].conic[0] = packed_conic_xy;
    splats[write_idx].conic[1] = packed_conic_z_pad;
    splats[write_idx].opacity = packed_opacity_pad;
    sort_depths[write_idx] = sort_key;
    sort_indices[write_idx] = write_idx;

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if ((write_idx % keys_per_dispatch) == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}