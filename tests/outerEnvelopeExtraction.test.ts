import { describe, expect, it } from "vitest";

import type {
  WebGpuFlexiCubesExtractionResult,
} from "../src/index.js";
import { ProxyOuterEnvelopeError } from "../src/index.js";
import { extractProxyOuterEnvelope } from
  "../src/pipeline/outerEnvelopeExtraction.js";
import { createCubeMesh } from "./fixtures/meshes.js";

const extractionAt = (
  halfExtent: number,
): WebGpuFlexiCubesExtractionResult => {
  const cube = createCubeMesh([0, 0, 0], halfExtent);
  return {
    mesh: {
      positions: cube.positions,
      indices: new Uint32Array(cube.indices),
      sourceCells: new Uint32Array(cube.positions.length / 3),
    },
    stats: {
      backend: "webgpu",
      surfaceCellCount: 8,
      surfaceEdgeCount: 12,
      boundarySurfaceEdgeCount: 0,
      dualVertexCount: 8,
      quadCount: 6,
      triangleCount: 12,
      memoryBytes: 0,
      elapsedMs: 0,
    },
  };
};

describe("extractProxyOuterEnvelope", () => {
  it("re-extracts at a larger iso value until containment passes", async () => {
    const requestedIsoValues: number[] = [];
    const result = await extractProxyOuterEnvelope(
      createCubeMesh(),
      0,
      {
        minimumSeparation: 0.01,
        maximumExpansion: 0.2,
        maximumAttempts: 3,
      },
      undefined,
      async isoValue => {
        requestedIsoValues.push(isoValue);
        return extractionAt(0.45 + isoValue);
      },
    );

    expect(requestedIsoValues).toHaveLength(2);
    expect(requestedIsoValues[0]).toBe(0);
    expect(requestedIsoValues[1]).toBeCloseTo(0.075);
    expect(result.evidence.initialVerification.violationCount)
      .toBeGreaterThan(0);
    expect(result.evidence.finalVerification.violationCount).toBe(0);
    expect(result.evidence.finalIsoValue).toBeCloseTo(0.075);
  });

  it("rejects when the declared expansion budget cannot contain the source", async () => {
    await expect(extractProxyOuterEnvelope(
      createCubeMesh(),
      0,
      {
        minimumSeparation: 0.01,
        maximumExpansion: 0.01,
        maximumAttempts: 2,
      },
      undefined,
      async isoValue => extractionAt(0.25 + isoValue),
    )).rejects.toBeInstanceOf(ProxyOuterEnvelopeError);
  });
});
