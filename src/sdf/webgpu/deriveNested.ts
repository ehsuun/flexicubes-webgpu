import type {
  GpuDenseScalarField3D,
  GridSize3,
} from "../../core/types.js";
import { cachedComputePipeline } from "../../webgpu/pipelines.js";
import { OwnedGpuDenseScalarField3D } from "./gpuField.js";
import { throwIfAborted } from "./runtime.js";

const WORKGROUP_SIZE = 64;
const PARAMETER_BYTES = 32;
const DEFAULT_MAXIMUM_BYTES = 256 * 1024 * 1024;

const DERIVE_NESTED_FIELD_WGSL = /* wgsl */ `
struct Params {
  fineSampleX: u32,
  fineSampleY: u32,
  coarseSampleX: u32,
  coarseSampleY: u32,
  coarseSampleZ: u32,
  cellRatio: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> fineValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> coarseValues: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let coarseIndex = id.x;
  let coarseCount =
    params.coarseSampleX * params.coarseSampleY * params.coarseSampleZ;
  if (coarseIndex >= coarseCount) {
    return;
  }
  let coarseX = coarseIndex % params.coarseSampleX;
  let coarseYZ = coarseIndex / params.coarseSampleX;
  let coarseY = coarseYZ % params.coarseSampleY;
  let coarseZ = coarseYZ / params.coarseSampleY;
  let fineX = coarseX * params.cellRatio;
  let fineY = coarseY * params.cellRatio;
  let fineZ = coarseZ * params.cellRatio;
  let fineIndex =
    fineX + params.fineSampleX * (fineY + params.fineSampleY * fineZ);
  coarseValues[coarseIndex] = fineValues[fineIndex];
}
`;

export type NestedFieldPlan = Readonly<{
  cellRatio: number;
  cellCounts: GridSize3;
  sampleCounts: GridSize3;
  sampleCount: number;
  byteLength: number;
}>;

export const resolveNestedFieldPlan = (
  fineCellCounts: GridSize3,
  cellRatio: number,
): NestedFieldPlan => {
  if (!Number.isSafeInteger(cellRatio) || cellRatio < 1) {
    throw new RangeError("nested field cellRatio must be a positive integer");
  }
  if (fineCellCounts.some(count => count % cellRatio !== 0)) {
    throw new RangeError(
      "fine field cell counts must be divisible by nested field cellRatio",
    );
  }
  const cellCounts: GridSize3 = [
    fineCellCounts[0] / cellRatio,
    fineCellCounts[1] / cellRatio,
    fineCellCounts[2] / cellRatio,
  ];
  const sampleCounts: GridSize3 = [
    cellCounts[0] + 1,
    cellCounts[1] + 1,
    cellCounts[2] + 1,
  ];
  const sampleCount = (
    sampleCounts[0] * sampleCounts[1] * sampleCounts[2]
  );
  return {
    cellRatio,
    cellCounts,
    sampleCounts,
    sampleCount,
    byteLength: sampleCount * Float32Array.BYTES_PER_ELEMENT,
  };
};

const parameterData = (
  field: GpuDenseScalarField3D,
  plan: NestedFieldPlan,
): Uint32Array => new Uint32Array([
  field.sampleCounts[0],
  field.sampleCounts[1],
  plan.sampleCounts[0],
  plan.sampleCounts[1],
  plan.sampleCounts[2],
  plan.cellRatio,
  0,
  0,
]);

export const deriveNestedDenseScalarFieldWebGpu = async (
  device: GPUDevice,
  fineField: GpuDenseScalarField3D,
  cellRatio: number,
  options: Readonly<{
    maxGpuBytes?: number;
    signal?: AbortSignal;
  }> = {},
): Promise<GpuDenseScalarField3D> => {
  if (fineField.disposed) {
    throw new Error("fine GPU scalar field has been disposed");
  }
  if (fineField.device !== device) {
    throw new Error("fine GPU scalar field belongs to a different GPUDevice");
  }
  const plan = resolveNestedFieldPlan(fineField.cellCounts, cellRatio);
  const maximumBytes = options.maxGpuBytes ?? DEFAULT_MAXIMUM_BYTES;
  if (plan.byteLength + PARAMETER_BYTES > maximumBytes) {
    throw new RangeError("nested GPU scalar field exceeds its memory budget");
  }
  throwIfAborted(options.signal);
  const output = device.createBuffer({
    label: `nested dense field ratio ${cellRatio}`,
    size: plan.byteLength,
    usage: GPUBufferUsage.STORAGE,
  });
  const parameters = device.createBuffer({
    label: `nested dense field ratio ${cellRatio} parameters`,
    size: PARAMETER_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  try {
    device.queue.writeBuffer(parameters, 0, parameterData(fineField, plan));
    const pipeline = await cachedComputePipeline(
      device,
      "sdf/derive-nested-field",
      "derive nested dense field",
      DERIVE_NESTED_FIELD_WGSL,
    );
    throwIfAborted(options.signal);
    const group = device.createBindGroup({
      label: `nested dense field ratio ${cellRatio}`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: fineField.buffer } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: parameters } },
      ],
    });
    const encoder = device.createCommandEncoder({
      label: `derive nested dense field ratio ${cellRatio}`,
    });
    const pass = encoder.beginComputePass({
      label: `derive nested dense field ratio ${cellRatio}`,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(plan.sampleCount / WORKGROUP_SIZE));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    throwIfAborted(options.signal);
    return new OwnedGpuDenseScalarField3D(
      device,
      {
        cellCounts: plan.cellCounts,
        sampleCounts: plan.sampleCounts,
        sampleCount: plan.sampleCount,
        sampleOrigin: fineField.sampleOrigin,
        sampleSpacing: [
          fineField.sampleSpacing[0] * cellRatio,
          fineField.sampleSpacing[1] * cellRatio,
          fineField.sampleSpacing[2] * cellRatio,
        ],
      },
      output,
      plan.byteLength,
      fineField.signConvention,
    );
  } catch (error) {
    output.destroy();
    throw error;
  } finally {
    parameters.destroy();
  }
};
