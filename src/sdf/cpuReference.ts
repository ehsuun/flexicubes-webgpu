import {
  createLattice3D,
  estimateDenseFieldMemoryBytes,
} from "../core/lattice.js";
import type {
  CpuSdfBakeResult,
  DenseScalarField3D,
  DenseSdfBakeOptions,
  IndexedMesh,
  ScalarFieldSignConvention,
  Vec3,
} from "../core/types.js";
import { classifyPointByParity } from "./parity.js";
import {
  pointTriangleDistanceSquared,
  prepareTriangles,
} from "./triangle.js";

export class SdfBakeAbortError extends Error {
  public override readonly name = "AbortError";

  public constructor() {
    super("SDF bake was aborted");
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw new SdfBakeAbortError();
  }
}

function validateSignPolicy(options: DenseSdfBakeOptions): void {
  if (
    options.signPolicy.kind === "shell"
    && (
      !Number.isFinite(options.signPolicy.halfThickness)
      || options.signPolicy.halfThickness <= 0
    )
  ) {
    throw new RangeError("shell halfThickness must be finite and positive");
  }
}

function signConvention(
  options: DenseSdfBakeOptions,
): ScalarFieldSignConvention {
  return options.signPolicy.kind === "unsigned"
    ? "unsigned"
    : "negative-inside";
}

/**
 * Deterministic correctness oracle.
 *
 * This deliberately performs brute-force distance and parity tests. It is for
 * fixtures and backend comparisons, not interactive production use.
 */
export function bakeDenseSdfCpuReference(
  mesh: IndexedMesh,
  options: DenseSdfBakeOptions,
): CpuSdfBakeResult {
  throwIfAborted(options.signal);
  validateSignPolicy(options);

  const triangles = prepareTriangles(mesh);
  const lattice = createLattice3D(options.domain, options.cellCounts);
  const values = new Float32Array(lattice.sampleCount);
  const sliceSampleCount = lattice.sampleCounts[0] * lattice.sampleCounts[1];
  const surfaceEpsilon = (
    Math.max(...lattice.sampleSpacing) * 1e-6
  );
  let parityRayTriangleTests = 0;
  let completedSamples = 0;

  options.onProgress?.({
    phase: "distance-and-sign",
    completed: completedSamples,
    total: lattice.sampleCount,
    fraction: 0,
  });

  const startTime = performance.now();
  for (let z = 0; z < lattice.sampleCounts[2]; z++) {
    for (let y = 0; y < lattice.sampleCounts[1]; y++) {
      for (let x = 0; x < lattice.sampleCounts[0]; x++) {
        if ((completedSamples & 255) === 0) {
          throwIfAborted(options.signal);
        }

        const point: Vec3 = [
          lattice.sampleOrigin[0] + x * lattice.sampleSpacing[0],
          lattice.sampleOrigin[1] + y * lattice.sampleSpacing[1],
          lattice.sampleOrigin[2] + z * lattice.sampleSpacing[2],
        ];
        let minimumDistanceSquared = Infinity;
        for (const triangle of triangles) {
          minimumDistanceSquared = Math.min(
            minimumDistanceSquared,
            pointTriangleDistanceSquared(point, triangle),
          );
        }

        const unsignedDistance = Math.sqrt(minimumDistanceSquared);
        let value = unsignedDistance;
        if (options.signPolicy.kind === "shell") {
          value -= options.signPolicy.halfThickness;
        } else if (
          options.signPolicy.kind === "parity"
          && unsignedDistance > surfaceEpsilon
        ) {
          const classification = classifyPointByParity(point, triangles);
          parityRayTriangleTests += classification.rayTriangleTests;
          if (classification.inside) {
            value = -value;
          }
        } else if (unsignedDistance <= surfaceEpsilon) {
          value = 0;
        }

        values[completedSamples] = value;
        completedSamples++;
      }
    }

    options.onProgress?.({
      phase: "distance-and-sign",
      completed: completedSamples,
      total: lattice.sampleCount,
      fraction: completedSamples / lattice.sampleCount,
    });
  }

  const field: DenseScalarField3D = {
    ...lattice,
    storage: "cpu-dense",
    layout: "x-fastest",
    values,
    signConvention: signConvention(options),
  };

  return {
    field,
    stats: {
      backend: "cpu-reference",
      triangleCount: triangles.length,
      sampleCount: lattice.sampleCount,
      memoryBytes: estimateDenseFieldMemoryBytes(lattice),
      distanceTests: lattice.sampleCount * triangles.length,
      parityRayTriangleTests,
      elapsedMs: performance.now() - startTime,
    },
  };
}
