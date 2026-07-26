import type {
  ExtractedMesh,
  FlexiCubesPhase,
  GpuDenseScalarField3D,
  WebGpuFlexiCubesExtractOptions,
  WebGpuFlexiCubesExtractionResult,
} from "../../core/types.js";
import { canonicalizeExtractedMesh } from "../canonicalize.js";
import {
  GpuBufferTracker,
  readBufferCopy,
} from "../../sdf/webgpu/runtime.js";
import {
  FLEXICUBES_AMBIGUITY_CHECKS,
  FLEXICUBES_DMC_TABLE,
  FLEXICUBES_DUAL_VERTEX_COUNTS,
} from "../tables.js";
import {
  BUILD_FLEXICUBES_DUAL_VERTICES_WGSL,
  CLASSIFY_FLEXICUBES_CELLS_WGSL,
  RESOLVE_FLEXICUBES_CASES_WGSL,
  TRIANGULATE_FLEXICUBES_WGSL,
} from "./shaders.js";
import { cachedComputePipeline } from "../../webgpu/pipelines.js";

const WORKGROUP_SIZE = 64;
const DEFAULT_BATCH_SIZE = 65_536;
const DEFAULT_MAX_GPU_BYTES = 256 * 1024 * 1024;

export async function prewarmFlexiCubesWebGpu(
  device: GPUDevice,
): Promise<void> {
  await Promise.all([
    cachedComputePipeline(
      device,
      "flexicubes/classify-cells",
      "classify FlexiCubes cells",
      CLASSIFY_FLEXICUBES_CELLS_WGSL,
    ),
    cachedComputePipeline(
      device,
      "flexicubes/resolve-cases",
      "resolve FlexiCubes cases",
      RESOLVE_FLEXICUBES_CASES_WGSL,
    ),
    cachedComputePipeline(
      device,
      "flexicubes/build-dual-vertices",
      "build FlexiCubes dual vertices",
      BUILD_FLEXICUBES_DUAL_VERTICES_WGSL,
    ),
    cachedComputePipeline(
      device,
      "flexicubes/triangulate",
      "triangulate FlexiCubes",
      TRIANGULATE_FLEXICUBES_WGSL,
    ),
  ]);
}

export class FlexiCubesGpuBudgetError extends Error {
  public override readonly name = "FlexiCubesGpuBudgetError";
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    const error = new Error("FlexiCubes extraction was aborted");
    error.name = "AbortError";
    throw error;
  }
}

function reportProgress(
  options: WebGpuFlexiCubesExtractOptions,
  phase: FlexiCubesPhase,
  completed: number,
  total: number,
): void {
  options.onProgress?.({
    phase,
    completed,
    total,
    fraction: total === 0 ? 1 : completed / total,
  });
}

function positiveInteger(
  name: string,
  value: number | undefined,
  fallback: number,
): number {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < 1) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return resolved;
}

function paramsData(
  field: GpuDenseScalarField3D,
  isoValue: number,
  workStart: number,
  workBatchCount: number,
  totalCells: number,
  maxQuads: number,
  totalEdges: number,
): Uint8Array {
  const buffer = new ArrayBuffer(96);
  const view = new DataView(buffer);
  view.setUint32(0, workStart, true);
  view.setUint32(4, workBatchCount, true);
  view.setUint32(8, totalCells, true);
  view.setUint32(12, maxQuads, true);
  view.setUint32(16, field.cellCounts[0], true);
  view.setUint32(20, field.cellCounts[1], true);
  view.setUint32(24, field.cellCounts[2], true);
  view.setUint32(28, totalEdges, true);
  view.setUint32(32, field.sampleCounts[0], true);
  view.setUint32(36, field.sampleCounts[1], true);
  view.setUint32(40, field.sampleCounts[2], true);
  view.setFloat32(48, isoValue, true);
  for (let axis = 0; axis < 3; axis++) {
    view.setFloat32(64 + axis * 4, field.sampleOrigin[axis]!, true);
    view.setFloat32(80 + axis * 4, field.sampleSpacing[axis]!, true);
  }
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

async function runBatches(
  device: GPUDevice,
  field: GpuDenseScalarField3D,
  options: WebGpuFlexiCubesExtractOptions,
  phase: FlexiCubesPhase,
  totalWork: number,
  batchSize: number,
  totalCells: number,
  maxQuads: number,
  totalEdges: number,
  paramsBuffer: GPUBuffer,
  pipeline: GPUComputePipeline,
  group: GPUBindGroup,
): Promise<void> {
  reportProgress(options, phase, 0, totalWork);
  for (
    let workStart = 0;
    workStart < totalWork;
    workStart += batchSize
  ) {
    throwIfAborted(options.signal);
    const workBatchCount = Math.min(batchSize, totalWork - workStart);
    device.queue.writeBuffer(
      paramsBuffer,
      0,
      paramsData(
        field,
        options.isoValue ?? 0,
        workStart,
        workBatchCount,
        totalCells,
        maxQuads,
        totalEdges,
      ),
    );
    const encoder = device.createCommandEncoder({
      label: `${phase} ${workStart}`,
    });
    const pass = encoder.beginComputePass({ label: phase });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(workBatchCount / WORKGROUP_SIZE));
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    reportProgress(
      options,
      phase,
      workStart + workBatchCount,
      totalWork,
    );
  }
}

function tableBuffer(
  device: GPUDevice,
  tracker: GpuBufferTracker,
  label: string,
  data: Int32Array | Uint32Array,
): GPUBuffer {
  const buffer = tracker.create(
    label,
    data.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    true,
  );
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function totalGridEdges(field: GpuDenseScalarField3D): number {
  return (
    field.cellCounts[0] * field.sampleCounts[1] * field.sampleCounts[2]
    + field.cellCounts[1] * field.sampleCounts[0] * field.sampleCounts[2]
    + field.cellCounts[2] * field.sampleCounts[0] * field.sampleCounts[1]
  );
}

function maximumInteriorQuads(field: GpuDenseScalarField3D): number {
  return (
    field.cellCounts[0]
      * Math.max(0, field.cellCounts[1] - 1)
      * Math.max(0, field.cellCounts[2] - 1)
    + field.cellCounts[1]
      * Math.max(0, field.cellCounts[0] - 1)
      * Math.max(0, field.cellCounts[2] - 1)
    + field.cellCounts[2]
      * Math.max(0, field.cellCounts[0] - 1)
      * Math.max(0, field.cellCounts[1] - 1)
  );
}

function emptyResult(
  surfaceCellCount: number,
  memoryBytes: number,
  startTime: number,
): WebGpuFlexiCubesExtractionResult {
  return {
    mesh: {
      positions: new Float32Array(),
      indices: new Uint32Array(),
      sourceCells: new Uint32Array(),
    },
    stats: {
      backend: "webgpu",
      surfaceCellCount,
      surfaceEdgeCount: 0,
      boundarySurfaceEdgeCount: 0,
      dualVertexCount: 0,
      quadCount: 0,
      triangleCount: 0,
      memoryBytes,
      elapsedMs: performance.now() - startTime,
    },
  };
}

export async function extractFlexiCubesWebGpu(
  device: GPUDevice,
  field: GpuDenseScalarField3D,
  options: WebGpuFlexiCubesExtractOptions = {},
): Promise<WebGpuFlexiCubesExtractionResult> {
  if (field.disposed) {
    throw new Error("GPU scalar field has been disposed");
  }
  if (field.device !== device) {
    throw new Error("GPU scalar field belongs to a different GPUDevice");
  }
  if (field.signConvention !== "negative-inside") {
    throw new RangeError(
      "FlexiCubes requires a negative-inside scalar field",
    );
  }
  const isoValue = options.isoValue ?? 0;
  if (!Number.isFinite(isoValue)) {
    throw new RangeError("isoValue must be finite");
  }
  const maxOutputTriangles = options.maxOutputTriangles ?? 10_000_000;
  if (
    !Number.isSafeInteger(maxOutputTriangles)
    || maxOutputTriangles < 0
  ) {
    throw new RangeError(
      "maxOutputTriangles must be a non-negative safe integer",
    );
  }
  if (
    device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE
    || device.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
  ) {
    throw new FlexiCubesGpuBudgetError(
      `device does not support ${WORKGROUP_SIZE}-thread workgroups`,
    );
  }
  if (device.limits.maxStorageBuffersPerShaderStage < 8) {
    throw new FlexiCubesGpuBudgetError(
      "device must support eight storage buffers per compute stage",
    );
  }

  const maxGpuBytes = positiveInteger(
    "maxGpuBytes",
    options.execution?.maxGpuBytes,
    DEFAULT_MAX_GPU_BYTES,
  );
  if (field.byteLength >= maxGpuBytes) {
    throw new FlexiCubesGpuBudgetError(
      "the input field already exceeds the declared GPU memory budget",
    );
  }
  const maximumBatchSize = (
    device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE
  );
  const cellBatchSize = Math.min(
    positiveInteger(
      "cellBatchSize",
      options.execution?.cellBatchSize,
      DEFAULT_BATCH_SIZE,
    ),
    maximumBatchSize,
  );
  const edgeBatchSize = Math.min(
    positiveInteger(
      "edgeBatchSize",
      options.execution?.edgeBatchSize,
      DEFAULT_BATCH_SIZE,
    ),
    maximumBatchSize,
  );
  const totalCells = (
    field.cellCounts[0] * field.cellCounts[1] * field.cellCounts[2]
  );
  const totalEdges = totalGridEdges(field);
  if (totalCells > 0xffff_ffff || totalEdges > 0xffff_ffff) {
    throw new FlexiCubesGpuBudgetError(
      "cell and edge counts must fit WebGPU u32 indexing",
    );
  }
  const maxQuads = Math.min(
    maximumInteriorQuads(field),
    Math.floor(maxOutputTriangles / 2),
  );
  const tracker = new GpuBufferTracker(
    device,
    maxGpuBytes - field.byteLength,
    (message) => new FlexiCubesGpuBudgetError(message),
  );
  const startTime = performance.now();

  try {
    throwIfAborted(options.signal);
    const paramsBuffer = tracker.create(
      "FlexiCubes params",
      96,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      true,
    );
    const rawCases = tracker.create(
      "raw FlexiCubes cases",
      totalCells * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
      true,
    );
    const resolvedCases = tracker.create(
      "resolved FlexiCubes cases",
      totalCells * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
      true,
    );
    const cellVertexOffsets = tracker.create(
      "FlexiCubes cell vertex offsets",
      totalCells * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE,
      true,
    );
    const ambiguityChecks = tableBuffer(
      device,
      tracker,
      "FlexiCubes ambiguity checks",
      Int32Array.from(FLEXICUBES_AMBIGUITY_CHECKS),
    );
    const dualVertexCounts = tableBuffer(
      device,
      tracker,
      "FlexiCubes dual vertex counts",
      Uint32Array.from(FLEXICUBES_DUAL_VERTEX_COUNTS),
    );
    const dmcTable = tableBuffer(
      device,
      tracker,
      "FlexiCubes DMC table",
      Int32Array.from(FLEXICUBES_DMC_TABLE),
    );
    const cellCounters = tracker.create(
      "FlexiCubes cell counters",
      2 * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE
        | GPUBufferUsage.COPY_SRC
        | GPUBufferUsage.COPY_DST,
      true,
    );
    const clearCounters = device.createCommandEncoder({
      label: "clear FlexiCubes cell counters",
    });
    clearCounters.clearBuffer(cellCounters);
    device.queue.submit([clearCounters.finish()]);

    const [classifyPipeline, resolvePipeline] = await Promise.all([
      cachedComputePipeline(
        device,
        "flexicubes/classify-cells",
        "classify FlexiCubes cells",
        CLASSIFY_FLEXICUBES_CELLS_WGSL,
      ),
      cachedComputePipeline(
        device,
        "flexicubes/resolve-cases",
        "resolve FlexiCubes cases",
        RESOLVE_FLEXICUBES_CASES_WGSL,
      ),
    ]);
    const classifyGroup = bindGroup(
      device,
      "classify FlexiCubes cells",
      classifyPipeline,
      [field.buffer, rawCases, paramsBuffer],
    );
    const resolveGroup = bindGroup(
      device,
      "resolve FlexiCubes cases",
      resolvePipeline,
      [
        rawCases,
        ambiguityChecks,
        dualVertexCounts,
        resolvedCases,
        cellVertexOffsets,
        cellCounters,
        paramsBuffer,
      ],
    );
    await runBatches(
      device,
      field,
      options,
      "classify-cells",
      totalCells,
      cellBatchSize,
      totalCells,
      maxQuads,
      totalEdges,
      paramsBuffer,
      classifyPipeline,
      classifyGroup,
    );
    await runBatches(
      device,
      field,
      options,
      "resolve-cases",
      totalCells,
      cellBatchSize,
      totalCells,
      maxQuads,
      totalEdges,
      paramsBuffer,
      resolvePipeline,
      resolveGroup,
    );

    const cellCounterValues = new Uint32Array(await readBufferCopy(
      device,
      tracker,
      cellCounters,
      2 * Uint32Array.BYTES_PER_ELEMENT,
      "FlexiCubes cell counters",
      options.signal,
    ));
    const dualVertexCount = cellCounterValues[0]!;
    const surfaceCellCount = cellCounterValues[1]!;
    tracker.destroy(cellCounters);
    tracker.destroy(rawCases);
    tracker.destroy(ambiguityChecks);
    if (dualVertexCount === 0) {
      return emptyResult(
        surfaceCellCount,
        field.byteLength + tracker.peakBytes,
        startTime,
      );
    }

    const outputPositions = tracker.create(
      "FlexiCubes positions",
      dualVertexCount * 3 * Float32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      true,
    );
    const sourceCells = tracker.create(
      "FlexiCubes source cells",
      dualVertexCount * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      true,
    );
    const dualPipeline = await cachedComputePipeline(
      device,
      "flexicubes/build-dual-vertices",
      "build FlexiCubes dual vertices",
      BUILD_FLEXICUBES_DUAL_VERTICES_WGSL,
    );
    const dualGroup = bindGroup(
      device,
      "build FlexiCubes dual vertices",
      dualPipeline,
      [
        field.buffer,
        resolvedCases,
        cellVertexOffsets,
        dmcTable,
        dualVertexCounts,
        outputPositions,
        sourceCells,
        paramsBuffer,
      ],
    );
    await runBatches(
      device,
      field,
      options,
      "build-dual-vertices",
      totalCells,
      cellBatchSize,
      totalCells,
      maxQuads,
      totalEdges,
      paramsBuffer,
      dualPipeline,
      dualGroup,
    );
    tracker.destroy(dualVertexCounts);

    const outputIndices = tracker.create(
      "FlexiCubes indices",
      maxQuads * 6 * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      true,
    );
    const edgeCounters = tracker.create(
      "FlexiCubes edge counters",
      3 * Uint32Array.BYTES_PER_ELEMENT,
      GPUBufferUsage.STORAGE
        | GPUBufferUsage.COPY_SRC
        | GPUBufferUsage.COPY_DST,
      true,
    );
    const clearEdgeCounters = device.createCommandEncoder({
      label: "clear FlexiCubes edge counters",
    });
    clearEdgeCounters.clearBuffer(edgeCounters);
    device.queue.submit([clearEdgeCounters.finish()]);

    const triangulatePipeline = await cachedComputePipeline(
      device,
      "flexicubes/triangulate",
      "triangulate FlexiCubes",
      TRIANGULATE_FLEXICUBES_WGSL,
    );
    const triangulateGroup = bindGroup(
      device,
      "triangulate FlexiCubes",
      triangulatePipeline,
      [
        field.buffer,
        resolvedCases,
        cellVertexOffsets,
        dmcTable,
        outputIndices,
        edgeCounters,
        paramsBuffer,
      ],
    );
    await runBatches(
      device,
      field,
      options,
      "triangulate",
      totalEdges,
      edgeBatchSize,
      totalCells,
      maxQuads,
      totalEdges,
      paramsBuffer,
      triangulatePipeline,
      triangulateGroup,
    );
    const edgeCounterValues = new Uint32Array(await readBufferCopy(
      device,
      tracker,
      edgeCounters,
      3 * Uint32Array.BYTES_PER_ELEMENT,
      "FlexiCubes edge counters",
      options.signal,
    ));
    const quadCount = edgeCounterValues[0]!;
    const overflowed = edgeCounterValues[1]! !== 0;
    const boundarySurfaceEdgeCount = edgeCounterValues[2]!;
    if (overflowed || quadCount > maxQuads) {
      throw new FlexiCubesGpuBudgetError(
        `FlexiCubes output exceeds ${maxOutputTriangles} triangles`,
      );
    }
    if (
      boundarySurfaceEdgeCount > 0
      && options.allowBoundaryOpen !== true
    ) {
      throw new RangeError(
        `isosurface reaches the lattice boundary at `
        + `${boundarySurfaceEdgeCount} surface edges`,
      );
    }
    tracker.destroy(edgeCounters);
    tracker.destroy(resolvedCases);
    tracker.destroy(cellVertexOffsets);
    tracker.destroy(dmcTable);
    tracker.destroy(paramsBuffer);

    reportProgress(options, "readback", 0, 3);
    const positions = new Float32Array(await readBufferCopy(
      device,
      tracker,
      outputPositions,
      dualVertexCount * 3 * Float32Array.BYTES_PER_ELEMENT,
      "FlexiCubes positions",
      options.signal,
    ));
    tracker.destroy(outputPositions);
    reportProgress(options, "readback", 1, 3);
    const provenance = new Uint32Array(await readBufferCopy(
      device,
      tracker,
      sourceCells,
      dualVertexCount * Uint32Array.BYTES_PER_ELEMENT,
      "FlexiCubes source cells",
      options.signal,
    ));
    tracker.destroy(sourceCells);
    reportProgress(options, "readback", 2, 3);
    const indices = new Uint32Array(await readBufferCopy(
      device,
      tracker,
      outputIndices,
      quadCount * 6 * Uint32Array.BYTES_PER_ELEMENT,
      "FlexiCubes indices",
      options.signal,
    ));
    tracker.destroy(outputIndices);
    reportProgress(options, "readback", 3, 3);

    const mesh: ExtractedMesh = canonicalizeExtractedMesh({
      positions,
      indices,
      sourceCells: provenance,
    });
    return {
      mesh,
      stats: {
        backend: "webgpu",
        surfaceCellCount,
        surfaceEdgeCount: quadCount + boundarySurfaceEdgeCount,
        boundarySurfaceEdgeCount,
        dualVertexCount,
        quadCount,
        triangleCount: quadCount * 2,
        memoryBytes: field.byteLength + tracker.peakBytes,
        elapsedMs: performance.now() - startTime,
      },
    };
  } finally {
    tracker.destroyAll();
  }
}
