import {
  bakeDenseSdfCpuReference,
  bakeDenseSdfWebGpu,
  bakeDenseSdfWebGpuResident,
  downloadDenseScalarFieldWebGpu,
  extractFlexiCubesCpuReference,
  extractFlexiCubesWebGpu,
  FlexiCubesGpuBudgetError,
  type DenseScalarField3D,
  type DenseSdfBakeOptions,
  type IndexedMesh,
} from "../../src/index.js";
import {
  createCubeMesh,
  createPlaneMesh,
} from "../fixtures/meshes.js";

interface FieldComparison {
  readonly maximumAbsoluteError: number;
  readonly signMismatchCount: number;
}

interface BrowserResult {
  readonly adapter: string;
  readonly cube: FieldComparison;
  readonly shell: FieldComparison;
  readonly composed: {
    readonly dualVertexCount: number;
    readonly triangleCount: number;
    readonly sdfMs: number;
    readonly extractionMs: number;
  };
  readonly millionTriangleCancellation: {
    readonly callReturnMs: number;
    readonly aborted: boolean;
  };
  readonly stress: {
    readonly triangleCount: number;
    readonly callReturnMs: number;
    readonly maximumTimerGapMs: number;
    readonly totalMs: number;
    readonly distanceReferences: number;
    readonly parityReferences: number;
    readonly peakBytes: number;
  } | null;
  readonly gpuStats: {
    readonly cubeMs: number;
    readonly shellMs: number;
    readonly peakBytes: number;
  };
}

function createMillionTriangleMesh(): IndexedMesh {
  const axisCount = 100;
  const triangleCount = axisCount ** 3;
  const positions = new Float32Array(triangleCount * 9);
  const indices = new Uint32Array(triangleCount * 3);
  const triangleRadius = 0.002;
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const x = triangle % axisCount;
    const y = Math.floor(triangle / axisCount) % axisCount;
    const z = Math.floor(triangle / (axisCount * axisCount));
    const centerX = -0.9 + 1.8 * (x + 0.5) / axisCount;
    const centerY = -0.9 + 1.8 * (y + 0.5) / axisCount;
    const centerZ = -0.9 + 1.8 * (z + 0.5) / axisCount;
    const positionOffset = triangle * 9;
    positions[positionOffset] = centerX - triangleRadius;
    positions[positionOffset + 1] = centerY - triangleRadius;
    positions[positionOffset + 2] = centerZ;
    positions[positionOffset + 3] = centerX + triangleRadius;
    positions[positionOffset + 4] = centerY - triangleRadius;
    positions[positionOffset + 5] = centerZ;
    positions[positionOffset + 6] = centerX;
    positions[positionOffset + 7] = centerY + triangleRadius;
    positions[positionOffset + 8] = centerZ;
    const indexOffset = triangle * 3;
    indices[indexOffset] = triangle * 3;
    indices[indexOffset + 1] = triangle * 3 + 1;
    indices[indexOffset + 2] = triangle * 3 + 2;
  }
  return { positions, indices };
}

async function runStressCase(device: GPUDevice): Promise<{
  readonly triangleCount: number;
  readonly callReturnMs: number;
  readonly maximumTimerGapMs: number;
  readonly totalMs: number;
  readonly distanceReferences: number;
  readonly parityReferences: number;
  readonly peakBytes: number;
}> {
  const mesh = createMillionTriangleMesh();
  let lastTimer = performance.now();
  let maximumTimerGapMs = 0;
  const timer = setInterval(() => {
    const now = performance.now();
    maximumTimerGapMs = Math.max(maximumTimerGapMs, now - lastTimer);
    lastTimer = now;
  }, 5);
  await new Promise((resolve) => setTimeout(resolve, 0));

  const totalStart = performance.now();
  const pending = bakeDenseSdfWebGpuResident(device, mesh, {
    domain: { min: [-1, -1, -1], max: [1, 1, 1] },
    cellCounts: [24, 24, 24],
    signPolicy: { kind: "parity" },
    execution: {
      distanceBinResolution: 12,
      triangleBatchSize: 65_536,
      sampleBatchSize: 4096,
      maxTriangleReferences: 12_000_000,
      maxGpuBytes: 512 * 1024 * 1024,
    },
  });
  const callReturnMs = performance.now() - totalStart;
  const result = await pending;
  const totalMs = performance.now() - totalStart;
  clearInterval(timer);
  result.field.dispose();

  if (callReturnMs >= 50) {
    throw new Error(
      `stress call blocked for ${callReturnMs}ms before yielding`,
    );
  }
  if (maximumTimerGapMs >= 50) {
    throw new Error(
      `stress path caused a ${maximumTimerGapMs}ms main-thread timer gap`,
    );
  }
  return {
    triangleCount: 1_000_000,
    callReturnMs,
    maximumTimerGapMs,
    totalMs,
    distanceReferences: result.stats.distanceTriangleReferences,
    parityReferences: result.stats.parityTriangleReferences,
    peakBytes: result.stats.memoryBytes,
  };
}

async function verifyMillionTriangleCancellation(
  device: GPUDevice,
): Promise<{ readonly callReturnMs: number; readonly aborted: boolean }> {
  const triangleCount = 1_000_000;
  const indices = new Uint32Array(triangleCount * 3);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const offset = triangle * 3;
    indices[offset] = 0;
    indices[offset + 1] = 1;
    indices[offset + 2] = 2;
  }
  const controller = new AbortController();
  const startTime = performance.now();
  const pending = bakeDenseSdfWebGpuResident(device, {
    positions: new Float32Array([
      -0.5, -0.5, 0,
      0.5, -0.5, 0,
      0, 0.5, 0,
    ]),
    indices,
  }, {
    domain: { min: [-1, -1, -1], max: [1, 1, 1] },
    cellCounts: [2, 2, 2],
    signPolicy: { kind: "shell", halfThickness: 0.1 },
    signal: controller.signal,
  });
  const callReturnMs = performance.now() - startTime;
  controller.abort();

  let aborted = false;
  try {
    const result = await pending;
    result.field.dispose();
  } catch (error) {
    aborted = error instanceof Error && error.name === "AbortError";
  }
  if (!aborted) {
    throw new Error("million-triangle preparation did not abort");
  }
  if (callReturnMs >= 50) {
    throw new Error(
      `million-triangle call blocked for ${callReturnMs}ms before yielding`,
    );
  }
  return { callReturnMs, aborted };
}

function positionRecords(
  positions: Float32Array,
  sourceCells: Uint32Array,
): string[] {
  const records: string[] = [];
  for (let index = 0; index < sourceCells.length; index++) {
    const offset = index * 3;
    records.push([
      sourceCells[index],
      positions[offset]!.toFixed(4),
      positions[offset + 1]!.toFixed(4),
      positions[offset + 2]!.toFixed(4),
    ].join(":"));
  }
  return records.sort();
}

function assertClosedManifold(indices: Uint32Array): void {
  const edgeCounts = new Map<string, number>();
  for (let index = 0; index < indices.length; index += 3) {
    const triangle = [
      indices[index]!,
      indices[index + 1]!,
      indices[index + 2]!,
    ];
    for (let edge = 0; edge < 3; edge++) {
      const start = triangle[edge]!;
      const end = triangle[(edge + 1) % 3]!;
      const key = start < end ? `${start}:${end}` : `${end}:${start}`;
      edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
    }
  }
  if (![...edgeCounts.values()].every((count) => count === 2)) {
    throw new Error("GPU extraction is not a closed two-manifold");
  }
}

function assertOutwardWinding(
  positions: Float32Array,
  indices: Uint32Array,
  center: readonly [number, number, number],
): void {
  for (let index = 0; index < indices.length; index += 3) {
    const aOffset = indices[index]! * 3;
    const bOffset = indices[index + 1]! * 3;
    const cOffset = indices[index + 2]! * 3;
    const ab = [
      positions[bOffset]! - positions[aOffset]!,
      positions[bOffset + 1]! - positions[aOffset + 1]!,
      positions[bOffset + 2]! - positions[aOffset + 2]!,
    ];
    const ac = [
      positions[cOffset]! - positions[aOffset]!,
      positions[cOffset + 1]! - positions[aOffset + 1]!,
      positions[cOffset + 2]! - positions[aOffset + 2]!,
    ];
    const normal = [
      ab[1]! * ac[2]! - ab[2]! * ac[1]!,
      ab[2]! * ac[0]! - ab[0]! * ac[2]!,
      ab[0]! * ac[1]! - ab[1]! * ac[0]!,
    ];
    const triangleCenter = [
      (
        positions[aOffset]!
        + positions[bOffset]!
        + positions[cOffset]!
      ) / 3,
      (
        positions[aOffset + 1]!
        + positions[bOffset + 1]!
        + positions[cOffset + 1]!
      ) / 3,
      (
        positions[aOffset + 2]!
        + positions[bOffset + 2]!
        + positions[cOffset + 2]!
      ) / 3,
    ];
    const outwardDot = (
      normal[0]! * (triangleCenter[0]! - center[0])
      + normal[1]! * (triangleCenter[1]! - center[1])
      + normal[2]! * (triangleCenter[2]! - center[2])
    );
    if (outwardDot <= 0) {
      throw new Error(`GPU triangle ${index / 3} is not wound outward`);
    }
  }
}

function compareFields(
  cpu: DenseScalarField3D,
  gpu: DenseScalarField3D,
): FieldComparison {
  if (cpu.values.length !== gpu.values.length) {
    throw new Error("CPU and GPU fields have different lengths");
  }

  let maximumAbsoluteError = 0;
  let signMismatchCount = 0;
  for (let index = 0; index < cpu.values.length; index++) {
    const cpuValue = cpu.values[index]!;
    const gpuValue = gpu.values[index]!;
    maximumAbsoluteError = Math.max(
      maximumAbsoluteError,
      Math.abs(cpuValue - gpuValue),
    );
    if (
      Math.abs(cpuValue) > 1e-5
      && Math.abs(gpuValue) > 1e-5
      && Math.sign(cpuValue) !== Math.sign(gpuValue)
    ) {
      signMismatchCount++;
    }
  }
  return { maximumAbsoluteError, signMismatchCount };
}

async function runCase(
  device: GPUDevice,
  mesh: IndexedMesh,
  options: DenseSdfBakeOptions,
): Promise<{
  readonly comparison: FieldComparison;
  readonly elapsedMs: number;
  readonly peakBytes: number;
}> {
  const cpu = bakeDenseSdfCpuReference(mesh, options);
  const gpu = await bakeDenseSdfWebGpu(device, mesh, {
    ...options,
    execution: {
      distanceBinResolution: 4,
      parityBinResolution: 4,
      triangleBatchSize: 64,
      sampleBatchSize: 128,
      maxGpuBytes: 32 * 1024 * 1024,
      maxTriangleReferences: 100_000,
    },
  });
  return {
    comparison: compareFields(cpu.field, gpu.field),
    elapsedMs: gpu.stats.elapsedMs,
    peakBytes: gpu.stats.memoryBytes,
  };
}

async function runComposedCase(device: GPUDevice): Promise<{
  readonly dualVertexCount: number;
  readonly triangleCount: number;
  readonly sdfMs: number;
  readonly extractionMs: number;
}> {
  const cubeCenter = [0.07, -0.04, 0.09] as const;
  const resident = await bakeDenseSdfWebGpuResident(
    device,
    createCubeMesh(cubeCenter, 0.43),
    {
      domain: { min: [-1, -1, -1], max: [1, 1, 1] },
      cellCounts: [8, 8, 8],
      signPolicy: { kind: "parity" },
      execution: {
        distanceBinResolution: 4,
        parityBinResolution: 4,
        maxGpuBytes: 64 * 1024 * 1024,
      },
    },
  );
  try {
    const cpuField = await downloadDenseScalarFieldWebGpu(
      device,
      resident.field,
    );
    const cpu = extractFlexiCubesCpuReference(cpuField);
    let budgetRefused = false;
    try {
      await extractFlexiCubesWebGpu(device, resident.field, {
        execution: {
          maxGpuBytes: resident.field.byteLength + 1,
        },
      });
    } catch (error) {
      budgetRefused = error instanceof FlexiCubesGpuBudgetError;
    }
    if (!budgetRefused) {
      throw new Error("FlexiCubes did not return its typed budget refusal");
    }
    const gpu = await extractFlexiCubesWebGpu(device, resident.field, {
      maxOutputTriangles: 100_000,
      execution: { maxGpuBytes: 64 * 1024 * 1024 },
    });

    if (gpu.stats.dualVertexCount !== cpu.stats.dualVertexCount) {
      throw new Error("CPU and GPU dual-vertex counts differ");
    }
    if (gpu.stats.triangleCount !== cpu.stats.triangleCount) {
      throw new Error("CPU and GPU triangle counts differ");
    }
    if (gpu.stats.boundarySurfaceEdgeCount !== 0) {
      throw new Error("composed proxy unexpectedly reaches the boundary");
    }
    const cpuPositions = positionRecords(
      cpu.mesh.positions,
      cpu.mesh.sourceCells,
    );
    const gpuPositions = positionRecords(
      gpu.mesh.positions,
      gpu.mesh.sourceCells,
    );
    if (JSON.stringify(cpuPositions) !== JSON.stringify(gpuPositions)) {
      const differences = cpuPositions.flatMap((record, index) => (
        record === gpuPositions[index]
          ? []
          : [`${index}: cpu=${record} gpu=${gpuPositions[index]}`]
      )).slice(0, 8);
      throw new Error(
        `CPU and GPU dual-vertex positions differ: ${differences.join("; ")}`,
      );
    }
    assertClosedManifold(gpu.mesh.indices);
    assertOutwardWinding(
      gpu.mesh.positions,
      gpu.mesh.indices,
      cubeCenter,
    );

    return {
      dualVertexCount: gpu.stats.dualVertexCount,
      triangleCount: gpu.stats.triangleCount,
      sdfMs: resident.stats.elapsedMs,
      extractionMs: gpu.stats.elapsedMs,
    };
  } finally {
    resident.field.dispose();
  }
}

async function main(): Promise<BrowserResult> {
  if (navigator.gpu === undefined) {
    throw new Error("navigator.gpu is unavailable");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) {
    throw new Error("no WebGPU adapter is available");
  }
  const device = await adapter.requestDevice();

  try {
    const cube = await runCase(device, createCubeMesh(), {
      domain: { min: [-1, -1, -1], max: [1, 1, 1] },
      cellCounts: [4, 4, 4],
      signPolicy: { kind: "parity" },
    });
    const shell = await runCase(device, createPlaneMesh(), {
      domain: { min: [-1, -1, -1], max: [1, 1, 1] },
      cellCounts: [4, 4, 4],
      signPolicy: { kind: "shell", halfThickness: 0.2 },
    });
    const composed = await runComposedCase(device);
    const millionTriangleCancellation = (
      await verifyMillionTriangleCancellation(device)
    );
    const stress = new URL(window.location.href).searchParams.has("stress")
      ? await runStressCase(device)
      : null;

    if (cube.comparison.maximumAbsoluteError > 1e-4) {
      throw new Error(
        `cube maximum error ${cube.comparison.maximumAbsoluteError} is too high`,
      );
    }
    if (cube.comparison.signMismatchCount !== 0) {
      throw new Error(
        `cube has ${cube.comparison.signMismatchCount} sign mismatches`,
      );
    }
    if (shell.comparison.maximumAbsoluteError > 1e-4) {
      throw new Error(
        `shell maximum error ${shell.comparison.maximumAbsoluteError} is too high`,
      );
    }

    const adapterInfo = adapter.info;
    return {
      adapter: [
        adapterInfo.vendor,
        adapterInfo.architecture,
        adapterInfo.device,
        adapterInfo.description,
      ].filter((value) => value.length > 0).join(" "),
      cube: cube.comparison,
      shell: shell.comparison,
      composed,
      millionTriangleCancellation,
      stress,
      gpuStats: {
        cubeMs: cube.elapsedMs,
        shellMs: shell.elapsedMs,
        peakBytes: Math.max(cube.peakBytes, shell.peakBytes),
      },
    };
  } finally {
    device.destroy();
  }
}

const resultElement = document.querySelector<HTMLPreElement>("#result");
if (resultElement === null) {
  throw new Error("result element is missing");
}

main().then(
  (result) => {
    resultElement.textContent = JSON.stringify(result);
    document.body.dataset.status = "passed";
  },
  (error: Error) => {
    resultElement.textContent = JSON.stringify({
      message: error.message,
      stack: error.stack,
    });
    document.body.dataset.status = "failed";
  },
);
