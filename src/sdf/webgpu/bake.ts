import type {
  DenseScalarField3D,
  IndexedMesh,
  ScalarFieldSignConvention,
  WebGpuDenseSdfBakeOptions,
  WebGpuResidentSdfBakeResult,
  WebGpuSdfBakeResult,
} from "../../core/types.js";
import { BAKE_DENSE_SDF_WGSL } from "./bakeShader.js";
import {
  COUNT_TRIANGLE_BINS_WGSL,
  SCATTER_DISTANCE_BINS_WGSL,
  SCATTER_PARITY_BINS_WGSL,
} from "./binningShaders.js";
import {
  resolveWebGpuSdfPlan,
  SdfGpuBudgetError,
  type ResolvedWebGpuSdfPlan,
} from "./planner.js";
import {
  GpuBufferTracker,
  prepareMeshForWebGpu,
  readBufferCopy,
  reportProgress,
  throwIfAborted,
  writeBufferChunked,
} from "./runtime.js";
import { OwnedGpuDenseScalarField3D } from "./gpuField.js";
import { cachedComputePipeline } from "../../webgpu/pipelines.js";

const WORKGROUP_SIZE = 64;

export async function prewarmSdfWebGpu(device: GPUDevice): Promise<void> {
  await Promise.all([
    cachedComputePipeline(
      device,
      "sdf/count-triangle-bins",
      "count triangle bins",
      COUNT_TRIANGLE_BINS_WGSL,
    ),
    cachedComputePipeline(
      device,
      "sdf/scatter-distance-bins",
      "scatter distance bins",
      SCATTER_DISTANCE_BINS_WGSL,
    ),
    cachedComputePipeline(
      device,
      "sdf/scatter-parity-bins",
      "scatter parity bins",
      SCATTER_PARITY_BINS_WGSL,
    ),
    cachedComputePipeline(
      device,
      "sdf/bake-dense",
      "bake dense SDF",
      BAKE_DENSE_SDF_WGSL,
    ),
  ]);
}

interface BinLayout {
  readonly offsets: Uint32Array;
  readonly headers: Uint32Array;
  readonly totalReferences: number;
}

interface UploadedMeshBuffers {
  readonly positions: GPUBuffer;
  readonly indices: GPUBuffer;
}

function prefixBinCounts(counts: Uint32Array): BinLayout {
  const offsets = new Uint32Array(counts.length);
  const headers = new Uint32Array(counts.length * 2);
  let totalReferences = 0;
  for (let index = 0; index < counts.length; index++) {
    offsets[index] = totalReferences;
    headers[index * 2] = totalReferences;
    headers[index * 2 + 1] = counts[index]!;
    totalReferences += counts[index]!;
    if (!Number.isSafeInteger(totalReferences) || totalReferences > 0xffff_ffff) {
      throw new SdfGpuBudgetError(
        "triangle-reference count exceeds WebGPU u32 indexing",
      );
    }
  }
  return { offsets, headers, totalReferences };
}

function countParams(
  plan: ResolvedWebGpuSdfPlan,
  options: WebGpuDenseSdfBakeOptions,
  triangleStart: number,
  triangleBatchCount: number,
): Uint8Array {
  const buffer = new ArrayBuffer(64);
  const view = new DataView(buffer);
  view.setUint32(0, triangleStart, true);
  view.setUint32(4, triangleBatchCount, true);
  view.setUint32(8, plan.triangleCount, true);
  view.setUint32(12, plan.distanceBinResolution, true);
  view.setUint32(16, plan.parityBinResolution, true);
  view.setUint32(20, plan.signMode, true);
  for (let axis = 0; axis < 3; axis++) {
    view.setFloat32(32 + axis * 4, options.domain.min[axis]!, true);
    view.setFloat32(48 + axis * 4, options.domain.max[axis]!, true);
  }
  return new Uint8Array(buffer);
}

function bakeParams(
  plan: ResolvedWebGpuSdfPlan,
  options: WebGpuDenseSdfBakeOptions,
  sampleStart: number,
  sampleBatchCount: number,
): Uint8Array {
  const buffer = new ArrayBuffer(112);
  const view = new DataView(buffer);
  view.setUint32(0, sampleStart, true);
  view.setUint32(4, sampleBatchCount, true);
  view.setUint32(8, plan.lattice.sampleCount, true);
  view.setUint32(12, plan.triangleCount, true);
  view.setUint32(16, plan.lattice.cellCounts[0], true);
  view.setUint32(20, plan.lattice.cellCounts[1], true);
  view.setUint32(24, plan.lattice.cellCounts[2], true);
  view.setUint32(28, plan.distanceBinResolution, true);
  view.setUint32(32, plan.parityBinResolution, true);
  view.setUint32(36, plan.signMode, true);
  for (let axis = 0; axis < 3; axis++) {
    view.setFloat32(48 + axis * 4, options.domain.min[axis]!, true);
    view.setFloat32(64 + axis * 4, options.domain.max[axis]!, true);
    view.setFloat32(80 + axis * 4, plan.lattice.sampleSpacing[axis]!, true);
  }
  view.setFloat32(96, plan.shellHalfThickness, true);
  view.setFloat32(100, plan.surfaceEpsilon, true);
  return new Uint8Array(buffer);
}

function bindGroup(
  device: GPUDevice,
  label: string,
  pipeline: GPUComputePipeline,
  buffers: readonly GPUBuffer[],
): GPUBindGroup {
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    })),
  });
}

function submitComputePass(
  device: GPUDevice,
  label: string,
  pipeline: GPUComputePipeline,
  group: GPUBindGroup,
  invocationCount: number,
  encoder: GPUCommandEncoder,
): void {
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(invocationCount / WORKGROUP_SIZE));
  pass.end();
}

async function uploadMesh(
  device: GPUDevice,
  tracker: GpuBufferTracker,
  plan: ResolvedWebGpuSdfPlan,
  mesh: Awaited<ReturnType<typeof prepareMeshForWebGpu>>,
  options: WebGpuDenseSdfBakeOptions,
): Promise<UploadedMeshBuffers> {
  const positions = tracker.create(
    "mesh positions",
    plan.positionBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    true,
  );
  const indices = tracker.create(
    "mesh indices",
    plan.indexBytes,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    true,
  );
  const totalElements = mesh.positions.length + mesh.indices.length;
  reportProgress(options, "upload", 0, totalElements);
  const completed = await writeBufferChunked(
    device,
    positions,
    mesh.positions,
    options,
    0,
    totalElements,
  );
  await writeBufferChunked(
    device,
    indices,
    mesh.indices,
    options,
    completed,
    totalElements,
  );
  return { positions, indices };
}

async function runTriangleBatches(
  device: GPUDevice,
  plan: ResolvedWebGpuSdfPlan,
  options: WebGpuDenseSdfBakeOptions,
  paramsBuffer: GPUBuffer,
  phase: "bin-count" | "bin-scatter",
  passes: readonly {
    readonly label: string;
    readonly pipeline: GPUComputePipeline;
    readonly group: GPUBindGroup;
  }[],
): Promise<void> {
  reportProgress(options, phase, 0, plan.triangleCount);
  for (
    let triangleStart = 0;
    triangleStart < plan.triangleCount;
    triangleStart += plan.triangleBatchSize
  ) {
    throwIfAborted(options.signal);
    const batchCount = Math.min(
      plan.triangleBatchSize,
      plan.triangleCount - triangleStart,
    );
    device.queue.writeBuffer(
      paramsBuffer,
      0,
      countParams(plan, options, triangleStart, batchCount),
    );
    const encoder = device.createCommandEncoder({
      label: `${phase} ${triangleStart}`,
    });
    for (const pass of passes) {
      submitComputePass(
        device,
        pass.label,
        pass.pipeline,
        pass.group,
        batchCount,
        encoder,
      );
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    reportProgress(
      options,
      phase,
      triangleStart + batchCount,
      plan.triangleCount,
    );
  }
}

async function runSampleBatches(
  device: GPUDevice,
  plan: ResolvedWebGpuSdfPlan,
  options: WebGpuDenseSdfBakeOptions,
  paramsBuffer: GPUBuffer,
  pipeline: GPUComputePipeline,
  group: GPUBindGroup,
): Promise<void> {
  reportProgress(options, "distance-and-sign", 0, plan.lattice.sampleCount);
  for (
    let sampleStart = 0;
    sampleStart < plan.lattice.sampleCount;
    sampleStart += plan.sampleBatchSize
  ) {
    throwIfAborted(options.signal);
    const batchCount = Math.min(
      plan.sampleBatchSize,
      plan.lattice.sampleCount - sampleStart,
    );
    device.queue.writeBuffer(
      paramsBuffer,
      0,
      bakeParams(plan, options, sampleStart, batchCount),
    );
    const encoder = device.createCommandEncoder({
      label: `distance-and-sign ${sampleStart}`,
    });
    submitComputePass(
      device,
      "distance and sign",
      pipeline,
      group,
      batchCount,
      encoder,
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    reportProgress(
      options,
      "distance-and-sign",
      sampleStart + batchCount,
      plan.lattice.sampleCount,
    );
  }
}

function signConvention(
  options: WebGpuDenseSdfBakeOptions,
): ScalarFieldSignConvention {
  return options.signPolicy.kind === "unsigned"
    ? "unsigned"
    : "negative-inside";
}

export async function bakeDenseSdfWebGpuResident(
  device: GPUDevice,
  mesh: IndexedMesh,
  options: WebGpuDenseSdfBakeOptions,
): Promise<WebGpuResidentSdfBakeResult> {
  const plan = resolveWebGpuSdfPlan(device, mesh, options);
  const tracker = new GpuBufferTracker(device, plan.maxGpuBytes);
  const startTime = performance.now();

  try {
    const preparedMesh = await prepareMeshForWebGpu(mesh, options);
    throwIfAborted(options.signal);
    const uploaded = await uploadMesh(
      device,
      tracker,
      plan,
      preparedMesh,
      options,
    );
    const output = tracker.create(
      "dense SDF output",
      plan.outputBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      true,
    );
    const countParamsBuffer = tracker.create(
      "binning params",
      64,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const distanceCounts = tracker.create(
      "distance bin counts",
      plan.distanceBinCount * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE
        | GPUBufferUsage.COPY_SRC
        | GPUBufferUsage.COPY_DST,
      true,
    );
    const parityCounts = tracker.create(
      "parity bin counts",
      plan.parityBinCount * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE
        | GPUBufferUsage.COPY_SRC
        | GPUBufferUsage.COPY_DST,
      true,
    );

    const clearEncoder = device.createCommandEncoder({
      label: "clear bin counts",
    });
    clearEncoder.clearBuffer(distanceCounts);
    clearEncoder.clearBuffer(parityCounts);
    device.queue.submit([clearEncoder.finish()]);

    const countPipeline = await cachedComputePipeline(
      device,
      "sdf/count-triangle-bins",
      "count triangle bins",
      COUNT_TRIANGLE_BINS_WGSL,
    );
    const countGroup = bindGroup(
      device,
      "count triangle bins",
      countPipeline,
      [
        uploaded.positions,
        uploaded.indices,
        distanceCounts,
        parityCounts,
        countParamsBuffer,
      ],
    );
    await runTriangleBatches(
      device,
      plan,
      options,
      countParamsBuffer,
      "bin-count",
      [{
        label: "count triangle bins",
        pipeline: countPipeline,
        group: countGroup,
      }],
    );

    const distanceCountsCopy = new Uint32Array(await readBufferCopy(
      device,
      tracker,
      distanceCounts,
      plan.distanceBinCount * Uint32Array.BYTES_PER_ELEMENT,
      "distance bin counts",
      options.signal,
    ));
    const parityCountsCopy = new Uint32Array(await readBufferCopy(
      device,
      tracker,
      parityCounts,
      plan.parityBinCount * Uint32Array.BYTES_PER_ELEMENT,
      "parity bin counts",
      options.signal,
    ));
    const distanceLayout = prefixBinCounts(distanceCountsCopy);
    const parityLayout = prefixBinCounts(parityCountsCopy);
    if (distanceLayout.totalReferences === 0) {
      throw new RangeError("mesh does not overlap the requested SDF domain");
    }
    const totalReferences = (
      distanceLayout.totalReferences + parityLayout.totalReferences
    );
    if (totalReferences > plan.maxTriangleReferences) {
      throw new SdfGpuBudgetError(
        `binning needs ${totalReferences} triangle references, exceeding the `
        + `${plan.maxTriangleReferences}-reference budget`,
      );
    }
    tracker.destroy(distanceCounts);
    tracker.destroy(parityCounts);

    const distanceOffsets = tracker.create(
      "distance bin offsets",
      distanceLayout.offsets.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const distanceCursors = tracker.create(
      "distance bin cursors",
      distanceLayout.offsets.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const distanceReferences = tracker.create(
      "distance triangle references",
      distanceLayout.totalReferences * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
      true,
    );
    const parityOffsets = tracker.create(
      "parity bin offsets",
      parityLayout.offsets.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const parityCursors = tracker.create(
      "parity bin cursors",
      parityLayout.offsets.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const parityReferences = tracker.create(
      "parity triangle references",
      parityLayout.totalReferences * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
      true,
    );
    device.queue.writeBuffer(distanceOffsets, 0, distanceLayout.offsets);
    device.queue.writeBuffer(parityOffsets, 0, parityLayout.offsets);
    const cursorClearEncoder = device.createCommandEncoder({
      label: "clear bin cursors",
    });
    cursorClearEncoder.clearBuffer(distanceCursors);
    cursorClearEncoder.clearBuffer(parityCursors);
    device.queue.submit([cursorClearEncoder.finish()]);

    const [
      scatterDistancePipeline,
      scatterParityPipeline,
    ] = await Promise.all([
      cachedComputePipeline(
        device,
        "sdf/scatter-distance-bins",
        "scatter distance bins",
        SCATTER_DISTANCE_BINS_WGSL,
      ),
      cachedComputePipeline(
        device,
        "sdf/scatter-parity-bins",
        "scatter parity bins",
        SCATTER_PARITY_BINS_WGSL,
      ),
    ]);
    const scatterDistanceGroup = bindGroup(
      device,
      "scatter distance bins",
      scatterDistancePipeline,
      [
        uploaded.positions,
        uploaded.indices,
        distanceOffsets,
        distanceCursors,
        distanceReferences,
        countParamsBuffer,
      ],
    );
    const scatterParityGroup = bindGroup(
      device,
      "scatter parity bins",
      scatterParityPipeline,
      [
        uploaded.positions,
        uploaded.indices,
        parityOffsets,
        parityCursors,
        parityReferences,
        countParamsBuffer,
      ],
    );
    await runTriangleBatches(
      device,
      plan,
      options,
      countParamsBuffer,
      "bin-scatter",
      [
        {
          label: "scatter distance bins",
          pipeline: scatterDistancePipeline,
          group: scatterDistanceGroup,
        },
        {
          label: "scatter parity bins",
          pipeline: scatterParityPipeline,
          group: scatterParityGroup,
        },
      ],
    );

    tracker.destroy(distanceOffsets);
    tracker.destroy(distanceCursors);
    tracker.destroy(parityOffsets);
    tracker.destroy(parityCursors);
    tracker.destroy(countParamsBuffer);

    const distanceHeaders = tracker.create(
      "distance bin headers",
      distanceLayout.headers.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const parityHeaders = tracker.create(
      "parity bin headers",
      parityLayout.headers.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const bakeParamsBuffer = tracker.create(
      "SDF bake params",
      112,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    device.queue.writeBuffer(distanceHeaders, 0, distanceLayout.headers);
    device.queue.writeBuffer(parityHeaders, 0, parityLayout.headers);

    const bakePipeline = await cachedComputePipeline(
      device,
      "sdf/bake-dense",
      "bake dense SDF",
      BAKE_DENSE_SDF_WGSL,
    );
    const bakeGroup = bindGroup(
      device,
      "bake dense SDF",
      bakePipeline,
      [
        uploaded.positions,
        uploaded.indices,
        distanceHeaders,
        distanceReferences,
        parityHeaders,
        parityReferences,
        output,
        bakeParamsBuffer,
      ],
    );
    await runSampleBatches(
      device,
      plan,
      options,
      bakeParamsBuffer,
      bakePipeline,
      bakeGroup,
    );

    const field = new OwnedGpuDenseScalarField3D(
      device,
      plan.lattice,
      output,
      plan.outputBytes,
      signConvention(options),
    );
    tracker.release(output);

    return {
      field,
      stats: {
        backend: "webgpu-dense-binned",
        triangleCount: plan.triangleCount,
        sampleCount: plan.lattice.sampleCount,
        memoryBytes: tracker.peakBytes,
        distanceBinResolution: plan.distanceBinResolution,
        parityBinResolution: plan.parityBinResolution,
        distanceTriangleReferences: distanceLayout.totalReferences,
        parityTriangleReferences: parityLayout.totalReferences,
        elapsedMs: performance.now() - startTime,
      },
    };
  } finally {
    tracker.destroyAll();
  }
}

export async function bakeDenseSdfWebGpu(
  device: GPUDevice,
  mesh: IndexedMesh,
  options: WebGpuDenseSdfBakeOptions,
): Promise<WebGpuSdfBakeResult> {
  const startTime = performance.now();
  const resident = await bakeDenseSdfWebGpuResident(device, mesh, options);
  const readbackTracker = new GpuBufferTracker(
    device,
    resident.field.byteLength,
  );
  try {
    reportProgress(options, "readback", 0, 1);
    const values = new Float32Array(await readBufferCopy(
      device,
      readbackTracker,
      resident.field.buffer,
      resident.field.byteLength,
      "dense SDF output",
      options.signal,
    ));
    reportProgress(options, "readback", 1, 1);
    const field: DenseScalarField3D = {
      cellCounts: resident.field.cellCounts,
      sampleCounts: resident.field.sampleCounts,
      sampleOrigin: resident.field.sampleOrigin,
      sampleSpacing: resident.field.sampleSpacing,
      sampleCount: resident.field.sampleCount,
      storage: "cpu-dense",
      layout: "x-fastest",
      values,
      signConvention: resident.field.signConvention,
    };
    return {
      field,
      stats: {
        ...resident.stats,
        memoryBytes: Math.max(
          resident.stats.memoryBytes,
          resident.field.byteLength * 2,
        ),
        elapsedMs: performance.now() - startTime,
      },
    };
  } finally {
    readbackTracker.destroyAll();
    resident.field.dispose();
  }
}
