import { describe, expect, it } from "vitest";

import { withAsyncResource } from
  "../src/pipeline/resourceLifetime.js";

describe("async resource lifetime", () => {
  it("releases after successful use", async () => {
    const released: string[] = [];
    const result = await withAsyncResource(
      async () => "fine-field",
      (resource) => {
        released.push(resource);
      },
      async () => "three-lods",
    );

    expect(result).toBe("three-lods");
    expect(released).toEqual(["fine-field"]);
  });

  it("releases after failure and cancellation", async () => {
    const released: string[] = [];
    await expect(withAsyncResource(
      async () => "derived-field",
      (resource) => {
        released.push(resource);
      },
      async () => {
        throw new Error("extraction failed");
      },
    )).rejects.toThrow("extraction failed");
    const aborted = new DOMException("aborted", "AbortError");
    await expect(withAsyncResource(
      async () => "fine-field",
      (resource) => {
        released.push(resource);
      },
      async () => {
        throw aborted;
      },
    )).rejects.toBe(aborted);

    expect(released).toEqual(["derived-field", "fine-field"]);
  });
});
