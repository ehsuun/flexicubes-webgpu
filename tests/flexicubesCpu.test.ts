import { describe, expect, it } from "vitest";
import {
  extractFlexiCubesCpuReference,
  FlexiCubesAbortError,
} from "../src/index.js";
import { FLEXICUBES_DUAL_VERTEX_COUNTS } from "../src/flexicubes/tables.js";
import {
  createAdjacentXCaseField,
  createAnalyticField,
  createSingleCaseField,
  createSphereField,
} from "./fixtures/fields.js";

function triangleOrientationDots(
  positions: Float32Array,
  indices: Uint32Array,
): number[] {
  const dots: number[] = [];
  for (let index = 0; index < indices.length; index += 3) {
    const aOffset = indices[index]! * 3;
    const bOffset = indices[index + 1]! * 3;
    const cOffset = indices[index + 2]! * 3;
    const ax = positions[aOffset]!;
    const ay = positions[aOffset + 1]!;
    const az = positions[aOffset + 2]!;
    const abx = positions[bOffset]! - ax;
    const aby = positions[bOffset + 1]! - ay;
    const abz = positions[bOffset + 2]! - az;
    const acx = positions[cOffset]! - ax;
    const acy = positions[cOffset + 1]! - ay;
    const acz = positions[cOffset + 2]! - az;
    const normalX = aby * acz - abz * acy;
    const normalY = abz * acx - abx * acz;
    const normalZ = abx * acy - aby * acx;
    const centerX = (
      ax + positions[bOffset]! + positions[cOffset]!
    ) / 3;
    const centerY = (
      ay + positions[bOffset + 1]! + positions[cOffset + 1]!
    ) / 3;
    const centerZ = (
      az + positions[bOffset + 2]! + positions[cOffset + 2]!
    ) / 3;
    dots.push(
      normalX * centerX + normalY * centerY + normalZ * centerZ,
    );
  }
  return dots;
}

function meshEdgeCounts(indices: Uint32Array): Map<string, number> {
  const counts = new Map<string, number>();
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
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}

describe("FlexiCubes CPU reference", () => {
  it("uses every upstream topology case and expected dual-vertex count", () => {
    for (let caseId = 0; caseId < 256; caseId++) {
      const result = extractFlexiCubesCpuReference(
        createSingleCaseField(caseId),
        { allowBoundaryOpen: true },
      );
      expect(
        result.mesh.positions.length / 3,
        `case ${caseId}`,
      ).toBe(FLEXICUBES_DUAL_VERTEX_COUNTS[caseId]);
    }
  });

  it("applies the upstream neighboring-cell ambiguity inversion", () => {
    const result = extractFlexiCubesCpuReference(
      createAdjacentXCaseField(61, 62),
      { allowBoundaryOpen: true },
    );

    // Raw cases 61 and 62 emit one dual vertex each. Their shared ambiguous
    // face inverts them to cases 194 and 193, which emit two each.
    expect(result.mesh.positions.length / 3).toBe(4);
  });

  it("extracts a deterministic closed sphere with outward winding", () => {
    const field = createSphereField(12);
    const first = extractFlexiCubesCpuReference(field);
    const second = extractFlexiCubesCpuReference(field);

    expect(first.stats.surfaceCellCount).toBeGreaterThan(0);
    expect(first.stats.boundarySurfaceEdgeCount).toBe(0);
    expect(first.stats.quadCount).toBeGreaterThan(0);
    expect(first.stats.triangleCount).toBe(first.stats.quadCount * 2);
    expect(first.mesh.positions).toEqual(second.mesh.positions);
    expect(first.mesh.indices).toEqual(second.mesh.indices);
    expect(first.mesh.sourceCells.length).toBe(
      first.mesh.positions.length / 3,
    );

    const orientationDots = triangleOrientationDots(
      first.mesh.positions,
      first.mesh.indices,
    );
    expect(Math.min(...orientationDots)).toBeGreaterThan(0);
    expect([...meshEdgeCounts(first.mesh.indices).values()].every(
      (count) => count === 2,
    )).toBe(true);
  });

  it("returns empty geometry for empty and fully occupied fields", () => {
    const empty = createAnalyticField(
      { min: [-1, -1, -1], max: [1, 1, 1] },
      [2, 2, 2],
      () => 1,
    );
    const full = createAnalyticField(
      { min: [-1, -1, -1], max: [1, 1, 1] },
      [2, 2, 2],
      () => -1,
    );

    expect(extractFlexiCubesCpuReference(empty).mesh.indices.length).toBe(0);
    expect(extractFlexiCubesCpuReference(full).mesh.indices.length).toBe(0);
  });

  it("rejects boundary-cut surfaces unless explicitly allowed", () => {
    const plane = createAnalyticField(
      { min: [-1, -1, -1], max: [1, 1, 1] },
      [4, 4, 4],
      ([x]) => x - 0.1,
    );

    expect(() => extractFlexiCubesCpuReference(plane)).toThrow(
      /lattice boundary/,
    );
    const open = extractFlexiCubesCpuReference(plane, {
      allowBoundaryOpen: true,
    });
    expect(open.stats.boundarySurfaceEdgeCount).toBeGreaterThan(0);
  });

  it("enforces the output triangle budget", () => {
    expect(() => extractFlexiCubesCpuReference(createSphereField(8), {
      maxOutputTriangles: 1,
    })).toThrow(/exceeds 1 triangles/);
  });

  it("rejects unsigned fields and respects cancellation", () => {
    const unsigned = createSphereField(4);
    const unsignedField = {
      ...unsigned,
      signConvention: "unsigned" as const,
    };
    expect(() => extractFlexiCubesCpuReference(unsignedField)).toThrow(
      /negative-inside/,
    );

    const controller = new AbortController();
    controller.abort();
    expect(() => extractFlexiCubesCpuReference(
      createSphereField(4),
      { signal: controller.signal },
    )).toThrow(FlexiCubesAbortError);
  });
});
