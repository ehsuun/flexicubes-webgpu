import { describe, expect, it } from "vitest";

import { canonicalizeExtractedMesh } from "../src/flexicubes/canonicalize.js";

describe("canonicalizeExtractedMesh", () => {
  it("removes vertex-allocation and triangle-emission ordering", () => {
    const first = canonicalizeExtractedMesh({
      positions: new Float32Array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 0,
        0, 0, 1,
      ]),
      sourceCells: new Uint32Array([3, 2, 1, 4]),
      indices: new Uint32Array([
        2, 0, 1,
        3, 2, 1,
      ]),
    });
    const reordered = canonicalizeExtractedMesh({
      positions: new Float32Array([
        0, 0, 1,
        0, 0, 0,
        0, 1, 0,
        1, 0, 0,
      ]),
      sourceCells: new Uint32Array([4, 1, 2, 3]),
      indices: new Uint32Array([
        2, 0, 1,
        2, 1, 3,
      ]),
    });

    expect(reordered.positions).toEqual(first.positions);
    expect(reordered.sourceCells).toEqual(first.sourceCells);
    expect(reordered.indices).toEqual(first.indices);
  });

  it("preserves each triangle's oriented cycle", () => {
    const result = canonicalizeExtractedMesh({
      positions: new Float32Array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
      ]),
      sourceCells: new Uint32Array([0, 1, 2]),
      indices: new Uint32Array([2, 0, 1]),
    });

    expect(result.indices).toEqual(new Uint32Array([0, 1, 2]));
  });
});
