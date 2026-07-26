import { describe, expect, it, vi } from "vitest";
import {
  bakeDenseSdfCpuReference,
  createLattice3D,
  latticeSampleIndex,
  SdfBakeAbortError,
} from "../src/index.js";
import {
  createCubeMesh,
  createPlaneMesh,
  mergeMeshes,
  reverseWinding,
} from "./fixtures/meshes.js";

const CUBE_DOMAIN = {
  min: [-1, -1, -1] as const,
  max: [1, 1, 1] as const,
};

describe("CPU SDF reference", () => {
  it("produces negative-inside vertex samples for a closed cube", () => {
    const result = bakeDenseSdfCpuReference(createCubeMesh(), {
      domain: CUBE_DOMAIN,
      cellCounts: [4, 4, 4],
      signPolicy: { kind: "parity" },
    });
    const lattice = createLattice3D(CUBE_DOMAIN, [4, 4, 4]);

    expect(result.field.values[
      latticeSampleIndex(lattice, 2, 2, 2)
    ]).toBeCloseTo(-0.5);
    expect(result.field.values[
      latticeSampleIndex(lattice, 0, 2, 2)
    ]).toBeCloseTo(0.5);
    expect(result.field.values[
      latticeSampleIndex(lattice, 1, 2, 2)
    ]).toBeCloseTo(0);
    expect(result.field.signConvention).toBe("negative-inside");
    expect(result.stats.triangleCount).toBe(12);
    expect(result.stats.sampleCount).toBe(125);
    expect(result.stats.memoryBytes).toBe(500);
    expect(result.stats.parityRayTriangleTests).toBeGreaterThan(0);
  });

  it("does not depend on triangle winding for parity sign", () => {
    const forward = bakeDenseSdfCpuReference(createCubeMesh(), {
      domain: CUBE_DOMAIN,
      cellCounts: [4, 4, 4],
      signPolicy: { kind: "parity" },
    });
    const reversed = bakeDenseSdfCpuReference(
      reverseWinding(createCubeMesh()),
      {
        domain: CUBE_DOMAIN,
        cellCounts: [4, 4, 4],
        signPolicy: { kind: "parity" },
      },
    );

    expect(reversed.field.values).toEqual(forward.field.values);
  });

  it("classifies disconnected closed components", () => {
    const mesh = mergeMeshes([
      createCubeMesh([-1, 0, 0], 0.25),
      createCubeMesh([1, 0, 0], 0.25),
    ]);
    const domain = {
      min: [-2, -1, -1] as const,
      max: [2, 1, 1] as const,
    };
    const result = bakeDenseSdfCpuReference(mesh, {
      domain,
      cellCounts: [8, 4, 4],
      signPolicy: { kind: "parity" },
    });
    const lattice = createLattice3D(domain, [8, 4, 4]);

    expect(result.field.values[
      latticeSampleIndex(lattice, 2, 2, 2)
    ]).toBeLessThan(0);
    expect(result.field.values[
      latticeSampleIndex(lattice, 6, 2, 2)
    ]).toBeLessThan(0);
    expect(result.field.values[
      latticeSampleIndex(lattice, 4, 2, 2)
    ]).toBeGreaterThan(0);
  });

  it("creates a two-sided shell around open geometry", () => {
    const result = bakeDenseSdfCpuReference(createPlaneMesh(), {
      domain: CUBE_DOMAIN,
      cellCounts: [2, 2, 4],
      signPolicy: { kind: "shell", halfThickness: 0.25 },
    });
    const lattice = createLattice3D(CUBE_DOMAIN, [2, 2, 4]);

    expect(result.field.values[
      latticeSampleIndex(lattice, 1, 1, 2)
    ]).toBeCloseTo(-0.25);
    expect(result.field.values[
      latticeSampleIndex(lattice, 1, 1, 3)
    ]).toBeCloseTo(0.25);
    expect(result.stats.parityRayTriangleTests).toBe(0);
  });

  it("supports unsigned distance without negative values", () => {
    const result = bakeDenseSdfCpuReference(createPlaneMesh(), {
      domain: CUBE_DOMAIN,
      cellCounts: [2, 2, 4],
      signPolicy: { kind: "unsigned" },
    });

    expect(result.field.signConvention).toBe("unsigned");
    expect(Math.min(...result.field.values)).toBe(0);
  });

  it("reports deterministic progress by completed sample count", () => {
    const onProgress = vi.fn();
    bakeDenseSdfCpuReference(createCubeMesh(), {
      domain: CUBE_DOMAIN,
      cellCounts: [2, 2, 2],
      signPolicy: { kind: "unsigned" },
      onProgress,
    });

    expect(onProgress).toHaveBeenCalledTimes(4);
    expect(onProgress.mock.calls[0]![0]).toMatchObject({
      completed: 0,
      total: 27,
      fraction: 0,
    });
    expect(onProgress.mock.calls.at(-1)![0]).toMatchObject({
      completed: 27,
      total: 27,
      fraction: 1,
    });
  });

  it("aborts before doing work", () => {
    const controller = new AbortController();
    controller.abort();

    expect(() => bakeDenseSdfCpuReference(createCubeMesh(), {
      domain: CUBE_DOMAIN,
      cellCounts: [4, 4, 4],
      signPolicy: { kind: "parity" },
      signal: controller.signal,
    })).toThrow(SdfBakeAbortError);
  });

  it("rejects a non-positive shell thickness", () => {
    expect(() => bakeDenseSdfCpuReference(createPlaneMesh(), {
      domain: CUBE_DOMAIN,
      cellCounts: [2, 2, 2],
      signPolicy: { kind: "shell", halfThickness: 0 },
    })).toThrow(/halfThickness/);
  });
});
