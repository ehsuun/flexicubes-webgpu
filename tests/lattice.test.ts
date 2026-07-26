import { describe, expect, it } from "vitest";
import {
  createLattice3D,
  estimateDenseFieldMemoryBytes,
  latticeSampleIndex,
  latticeSamplePosition,
} from "../src/index.js";

describe("lattice contracts", () => {
  it("stores values at cell corners including both domain endpoints", () => {
    const lattice = createLattice3D(
      { min: [-1, -2, -3], max: [1, 4, 5] },
      [2, 3, 4],
    );

    expect(lattice.cellCounts).toEqual([2, 3, 4]);
    expect(lattice.sampleCounts).toEqual([3, 4, 5]);
    expect(lattice.sampleCount).toBe(60);
    expect(lattice.sampleSpacing).toEqual([1, 2, 2]);
    expect(latticeSamplePosition(lattice, 0, 0, 0)).toEqual([-1, -2, -3]);
    expect(latticeSamplePosition(lattice, 2, 3, 4)).toEqual([1, 4, 5]);
  });

  it("uses x-fastest linear indexing", () => {
    const lattice = createLattice3D(
      { min: [0, 0, 0], max: [2, 2, 2] },
      [2, 2, 2],
    );

    expect(latticeSampleIndex(lattice, 0, 0, 0)).toBe(0);
    expect(latticeSampleIndex(lattice, 1, 0, 0)).toBe(1);
    expect(latticeSampleIndex(lattice, 0, 1, 0)).toBe(3);
    expect(latticeSampleIndex(lattice, 0, 0, 1)).toBe(9);
    expect(latticeSampleIndex(lattice, 2, 2, 2)).toBe(26);
    expect(estimateDenseFieldMemoryBytes(lattice)).toBe(27 * 4);
  });

  it("rejects invalid domains, cell counts, and coordinates", () => {
    expect(() => createLattice3D(
      { min: [0, 0, 0], max: [0, 1, 1] },
      [1, 1, 1],
    )).toThrow(/domain\.max\[0\]/);
    expect(() => createLattice3D(
      { min: [0, 0, 0], max: [1, 1, 1] },
      [0, 1, 1],
    )).toThrow(/cellCounts\[0\]/);

    const lattice = createLattice3D(
      { min: [0, 0, 0], max: [1, 1, 1] },
      [1, 1, 1],
    );
    expect(() => latticeSampleIndex(lattice, 2, 0, 0)).toThrow(
      /outside the lattice/,
    );
  });
});
