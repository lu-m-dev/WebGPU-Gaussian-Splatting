import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (v: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data as ArrayBuffer);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  const draw_indirect_buffer = createBuffer(
    device,
    'draw indirect buffer',
    4 * Uint32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    new Uint32Array([6, pc.num_points, 0, 0])
  );

  const SPLAT_SIZE = 32;
  const splat_buffer = createBuffer(
    device,
    'splats buffer',
    pc.num_points * SPLAT_SIZE,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX
  );

  const settings_buffer = createBuffer(
    device,
    'render settings buffer',
    4 * Float32Array.BYTES_PER_ELEMENT,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    new Float32Array([1.0, pc.sh_deg, pc.num_points, 0.0])
  );

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const preprocess_bind_group0 = device.createBindGroup({
    label: 'preprocess 0',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });

  const preprocess_bind_group1 = device.createBindGroup({
    label: 'preprocess 1',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: settings_buffer } },
    ],
  });

  const render_shader = device.createShaderModule({ code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'gaussian camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const render_bind_group = device.createBindGroup({
    label: 'gaussian splats',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 2, resource: { buffer: settings_buffer } },
    ],
  });
  

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  let gaussian_multiplier = 1.0;

  function set_gaussian_multiplier(v: number) {
    gaussian_multiplier = v;
    device.queue.writeBuffer(settings_buffer, 0, new Float32Array([gaussian_multiplier, pc.sh_deg, pc.num_points, 0.0]));
  }

  function record_preprocess(encoder: GPUCommandEncoder) {
    device.queue.writeBuffer(sorter.sort_info_buffer, 0, new Uint32Array([0, 0, 0, 0, 0]));
    device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, new Uint32Array([0, 1, 1]));
    device.queue.writeBuffer(settings_buffer, 0, new Float32Array([gaussian_multiplier, pc.sh_deg, pc.num_points, 0.0]));

    const pass = encoder.beginComputePass({ label: 'preprocess' });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, preprocess_bind_group0);
    pass.setBindGroup(1, preprocess_bind_group1);
    pass.setBindGroup(2, sort_bind_group);
    const wg = Math.ceil(pc.num_points / C.histogram_wg_size);
    pass.dispatchWorkgroups(wg);
    pass.end();
  }

  function record_render(encoder: GPUCommandEncoder, texture_view: GPUTextureView) {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, render_bind_group);
    pass.drawIndirect(draw_indirect_buffer, 0);
    pass.end();
  }
  

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      record_preprocess(encoder);
      sorter.sort(encoder);
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, draw_indirect_buffer, 4, 4);
      record_render(encoder, texture_view);
    },
    camera_buffer,
    set_gaussian_multiplier,
  };
}
