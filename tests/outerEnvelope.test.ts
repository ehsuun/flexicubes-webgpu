import { describe, expect, it } from "vitest";

import {
  verifyProxyOuterEnvelope,
} from "../src/index.js";
import { createCubeMesh } from "./fixtures/meshes.js";

const extractedCube = (halfExtent: number) => {
  const cube = createCubeMesh([0, 0, 0], halfExtent);
  return {
    positions: cube.positions,
    indices: new Uint32Array(cube.indices),
    sourceCells: new Uint32Array(cube.positions.length / 3),
  };
};

describe("verifyProxyOuterEnvelope", () => {
  it("reports where a proxy cuts inside sampled source surfaces", async () => {
    const evidence = await verifyProxyOuterEnvelope(
      createCubeMesh(),
      extractedCube(0.45),
      {
        minimumSeparation: 0.01,
        maximumSourceSamples: 128,
      },
    );

    expect(evidence.method).toBe("sampled-source-surface");
    expect(evidence.violationCount).toBeGreaterThan(0);
    expect(evidence.minimumSignedSeparation).toBeCloseTo(-0.05);
    expect(evidence.maximumIngress).toBeCloseTo(0.06);
  });

  it("accepts a sampled source surface inside a conservative proxy", async () => {
    const evidence = await verifyProxyOuterEnvelope(
      createCubeMesh(),
      extractedCube(0.55),
      {
        minimumSeparation: 0.01,
      },
    );

    expect(evidence.violationCount).toBe(0);
    expect(evidence.minimumSignedSeparation).toBeCloseTo(0.05);
  });

  it("is deterministic under a bounded sample budget", async () => {
    const first = await verifyProxyOuterEnvelope(
      createCubeMesh(),
      extractedCube(0.45),
      {
        minimumSeparation: 0.01,
        maximumSourceSamples: 9,
      },
    );
    const second = await verifyProxyOuterEnvelope(
      createCubeMesh(),
      extractedCube(0.45),
      {
        minimumSeparation: 0.01,
        maximumSourceSamples: 9,
      },
    );

    expect(second).toEqual(first);
    expect(first.sourceSampleCount).toBe(9);
  });
});
