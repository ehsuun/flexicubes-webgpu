import {
  createLattice3D,
  estimateDenseFieldMemoryBytes,
} from "../../core/lattice.js";
import type {
  IndexedMesh,
  Lattice3D,
  WebGpuDenseSdfBakeOptions,
} from "../../core/types.js";

const DEFAULT_MAX_GPU_BYTES = 256 * 1024 * 1024;
const DEFAULT_MAX_TRIANGLE_REFERENCES = 32 * 1024 * 1024;
const DEFAULT_BATCH_SIZE = 65_536;
const WORKGROUP_SIZE = 64;

export class SdfGpuBudgetError extends Error {
  public override readonly name = "SdfGpuBudgetError";
}

export interface ResolvedWebGpuSdfPlan {
  readonly lattice: Lattice3D;
  readonly triangleCount: number;
  readonly positionBytes: number;
  readonly indexBytes: number;
  readonly outputBytes: number;
  readonly distanceBinResolution: number;
  readonly distanceBinCount: number;
  readonly parityBinResolution: number;
  readonly parityBinCount: number;
  readonly triangleBatchSize: number;
  readonly sampleBatchSize: number;
  readonly maxGpuBytes: number;
  readonly maxTriangleReferences: number;
  readonly surfaceEpsilon: number;
  readonly signMode: 0 | 1 | 2;
  readonly shellHalfThickness: number;
}

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
}

function resolvePositiveInteger(
  name: string,
  value: number | undefined,
  fallback: number,
): number {
  const resolved = value ?? fallback;
  assertPositiveInteger(name, resolved);
  return resolved;
}

function chooseBinResolution(lattice: Lattice3D): number {
  const maximumCells = Math.max(...lattice.cellCounts);
  if (maximumCells <= 32) {
    return 8;
  }
  if (maximumCells <= 64) {
    return 12;
  }
  return 16;
}

function signMode(options: WebGpuDenseSdfBakeOptions): 0 | 1 | 2 {
  if (options.signPolicy.kind === "parity") {
    return 1;
  }
  if (options.signPolicy.kind === "shell") {
    return 2;
  }
  return 0;
}

function validateMeshShape(mesh: IndexedMesh): void {
  if (mesh.positions.length === 0 || mesh.positions.length % 3 !== 0) {
    throw new RangeError(
      "positions must contain one or more tightly packed XYZ vertices",
    );
  }
  if (mesh.indices.length === 0 || mesh.indices.length % 3 !== 0) {
    throw new RangeError(
      "indices must contain one or more complete triangles",
    );
  }
}

function assertStorageBindingFits(
  name: string,
  bytes: number,
  device: GPUDevice,
): void {
  if (bytes > device.limits.maxStorageBufferBindingSize) {
    throw new SdfGpuBudgetError(
      `${name} requires ${bytes} bytes, exceeding the device storage `
      + `binding limit of ${device.limits.maxStorageBufferBindingSize}`,
    );
  }
  if (bytes > device.limits.maxBufferSize) {
    throw new SdfGpuBudgetError(
      `${name} requires ${bytes} bytes, exceeding the device buffer limit `
      + `of ${device.limits.maxBufferSize}`,
    );
  }
}

export function resolveWebGpuSdfPlan(
  device: GPUDevice,
  mesh: IndexedMesh,
  options: WebGpuDenseSdfBakeOptions,
): ResolvedWebGpuSdfPlan {
  validateMeshShape(mesh);
  const lattice = createLattice3D(options.domain, options.cellCounts);
  const triangleCount = mesh.indices.length / 3;
  if (triangleCount > 0xffff_ffff || lattice.sampleCount > 0xffff_ffff) {
    throw new SdfGpuBudgetError(
      "triangle and sample counts must fit in WebGPU u32 indexing",
    );
  }
  if (
    device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE
    || device.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
  ) {
    throw new SdfGpuBudgetError(
      `device does not support ${WORKGROUP_SIZE}-thread compute workgroups`,
    );
  }
  if (device.limits.maxStorageBuffersPerShaderStage < 8) {
    throw new SdfGpuBudgetError(
      "device must support eight storage buffers per compute stage",
    );
  }

  const requestedDistanceBins = (
    options.execution?.distanceBinResolution
    ?? chooseBinResolution(lattice)
  );
  const requestedParityBins = (
    options.execution?.parityBinResolution
    ?? requestedDistanceBins
  );
  const distanceBinResolution = resolvePositiveInteger(
    "distanceBinResolution",
    requestedDistanceBins,
    8,
  );
  const parityBinResolution = resolvePositiveInteger(
    "parityBinResolution",
    requestedParityBins,
    8,
  );
  if (distanceBinResolution > 64 || parityBinResolution > 64) {
    throw new RangeError("bin resolutions must not exceed 64");
  }

  const maximumBatchSize = (
    device.limits.maxComputeWorkgroupsPerDimension * WORKGROUP_SIZE
  );
  const triangleBatchSize = Math.min(
    resolvePositiveInteger(
      "triangleBatchSize",
      options.execution?.triangleBatchSize,
      DEFAULT_BATCH_SIZE,
    ),
    maximumBatchSize,
  );
  const sampleBatchSize = Math.min(
    resolvePositiveInteger(
      "sampleBatchSize",
      options.execution?.sampleBatchSize,
      DEFAULT_BATCH_SIZE,
    ),
    maximumBatchSize,
  );
  const maxGpuBytes = resolvePositiveInteger(
    "maxGpuBytes",
    options.execution?.maxGpuBytes,
    DEFAULT_MAX_GPU_BYTES,
  );
  const maxTriangleReferences = resolvePositiveInteger(
    "maxTriangleReferences",
    options.execution?.maxTriangleReferences,
    DEFAULT_MAX_TRIANGLE_REFERENCES,
  );
  const positionBytes = mesh.positions.byteLength;
  const indexBytes = mesh.indices.length * Uint32Array.BYTES_PER_ELEMENT;
  const outputBytes = estimateDenseFieldMemoryBytes(lattice);
  assertStorageBindingFits("positions", positionBytes, device);
  assertStorageBindingFits("indices", indexBytes, device);
  assertStorageBindingFits("SDF output", outputBytes, device);

  const distanceBinCount = (
    distanceBinResolution
    * distanceBinResolution
    * distanceBinResolution
  );
  const parityBinCount = (
    3
    * parityBinResolution
    * parityBinResolution
  );
  const countBytes = (
    distanceBinCount + parityBinCount
  ) * Uint32Array.BYTES_PER_ELEMENT;
  const baseBytes = positionBytes + indexBytes + outputBytes + countBytes;
  if (baseBytes > maxGpuBytes) {
    throw new SdfGpuBudgetError(
      `base GPU allocation requires ${baseBytes} bytes, exceeding the `
      + `declared ${maxGpuBytes}-byte budget`,
    );
  }

  const shellHalfThickness = options.signPolicy.kind === "shell"
    ? options.signPolicy.halfThickness
    : 0;
  if (
    options.signPolicy.kind === "shell"
    && (
      !Number.isFinite(shellHalfThickness)
      || shellHalfThickness <= 0
    )
  ) {
    throw new RangeError("shell halfThickness must be finite and positive");
  }

  return {
    lattice,
    triangleCount,
    positionBytes,
    indexBytes,
    outputBytes,
    distanceBinResolution,
    distanceBinCount,
    parityBinResolution,
    parityBinCount,
    triangleBatchSize,
    sampleBatchSize,
    maxGpuBytes,
    maxTriangleReferences,
    surfaceEpsilon: Math.max(...lattice.sampleSpacing) * 1e-5,
    signMode: signMode(options),
    shellHalfThickness,
  };
}
