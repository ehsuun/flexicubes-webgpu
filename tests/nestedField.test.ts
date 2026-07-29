import { describe, expect, it } from "vitest";

import { resolveNestedFieldPlan } from "../src/sdf/webgpu/deriveNested.js";

describe("nested dense field plan", () => {
  it("keeps coarse samples aligned to integer-ratio fine samples", () => {
    expect(resolveNestedFieldPlan([24, 16, 8], 4)).toEqual({
      cellRatio: 4,
      cellCounts: [6, 4, 2],
      sampleCounts: [7, 5, 3],
      sampleCount: 105,
      byteLength: 420,
    });
  });

  it("rejects fractional, zero, and non-divisible ratios", () => {
    expect(() => resolveNestedFieldPlan([24, 16, 8], 0)).toThrow(
      /positive integer/,
    );
    expect(() => resolveNestedFieldPlan([24, 16, 8], 1.5)).toThrow(
      /positive integer/,
    );
    expect(() => resolveNestedFieldPlan([24, 16, 8], 3)).toThrow(
      /divisible/,
    );
  });
});
