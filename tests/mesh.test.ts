import { describe, expect, it } from "vitest";
import {
  computeMeshBounds,
  fitSdfDomain,
  transformIndexedMesh,
  validateIndexedMesh,
} from "../src/index.js";
import { createCubeMesh, createPlaneMesh } from "./fixtures/meshes.js";

describe("indexed mesh helpers", () => {
  it("computes bounds and pads degenerate axes", () => {
    expect(computeMeshBounds(createCubeMesh())).toEqual({
      min: [-0.5, -0.5, -0.5],
      max: [0.5, 0.5, 0.5],
    });

    const planeDomain = fitSdfDomain(createPlaneMesh(), 0.1, 0.05);
    expect(planeDomain.min[0]).toBeCloseTo(-0.9);
    expect(planeDomain.min[1]).toBeCloseTo(-0.9);
    expect(planeDomain.min[2]).toBeCloseTo(-0.15);
    expect(planeDomain.max[0]).toBeCloseTo(0.9);
    expect(planeDomain.max[1]).toBeCloseTo(0.9);
    expect(planeDomain.max[2]).toBeCloseTo(0.15);
  });

  it("applies a renderer-neutral column-major transform", () => {
    const transformed = transformIndexedMesh(createCubeMesh(), [
      2, 0, 0, 0,
      0, 3, 0, 0,
      0, 0, 4, 0,
      5, 6, 7, 1,
    ]);

    expect(computeMeshBounds(transformed)).toEqual({
      min: [4, 4.5, 5],
      max: [6, 7.5, 9],
    });
  });

  it("rejects incomplete and out-of-range mesh data", () => {
    expect(() => validateIndexedMesh({
      positions: new Float32Array([0, 0]),
      indices: new Uint16Array([0, 0, 0]),
    })).toThrow(/positions/);
    expect(() => validateIndexedMesh({
      positions: new Float32Array([0, 0, 0]),
      indices: new Uint16Array([0, 1, 0]),
    })).toThrow(/missing vertex/);
  });
});
