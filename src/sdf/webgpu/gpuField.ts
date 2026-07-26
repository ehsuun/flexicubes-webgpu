import type {
  DenseScalarField3D,
  GpuDenseScalarField3D,
  Lattice3D,
  ScalarFieldSignConvention,
} from "../../core/types.js";
import {
  GpuBufferTracker,
  readBufferCopy,
} from "./runtime.js";

export class OwnedGpuDenseScalarField3D
implements GpuDenseScalarField3D {
  public readonly storage = "gpu-dense";
  public readonly layout = "x-fastest";
  public readonly cellCounts;
  public readonly sampleCounts;
  public readonly sampleOrigin;
  public readonly sampleSpacing;
  public readonly sampleCount;
  public readonly device;
  public readonly buffer;
  public readonly byteLength;
  public readonly signConvention;
  #disposed = false;

  public constructor(
    device: GPUDevice,
    lattice: Lattice3D,
    buffer: GPUBuffer,
    byteLength: number,
    signConvention: ScalarFieldSignConvention,
  ) {
    this.device = device;
    this.cellCounts = lattice.cellCounts;
    this.sampleCounts = lattice.sampleCounts;
    this.sampleOrigin = lattice.sampleOrigin;
    this.sampleSpacing = lattice.sampleSpacing;
    this.sampleCount = lattice.sampleCount;
    this.buffer = buffer;
    this.byteLength = byteLength;
    this.signConvention = signConvention;
  }

  public get disposed(): boolean {
    return this.#disposed;
  }

  public dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.buffer.destroy();
    this.#disposed = true;
  }
}

export async function downloadDenseScalarFieldWebGpu(
  device: GPUDevice,
  field: GpuDenseScalarField3D,
  signal?: AbortSignal,
): Promise<DenseScalarField3D> {
  if (field.disposed) {
    throw new Error("GPU scalar field has been disposed");
  }
  if (field.device !== device) {
    throw new Error("GPU scalar field belongs to a different GPUDevice");
  }
  const tracker = new GpuBufferTracker(device, field.byteLength);
  try {
    const values = new Float32Array(await readBufferCopy(
      device,
      tracker,
      field.buffer,
      field.byteLength,
      "dense scalar field",
      signal,
    ));
    return {
      cellCounts: field.cellCounts,
      sampleCounts: field.sampleCounts,
      sampleOrigin: field.sampleOrigin,
      sampleSpacing: field.sampleSpacing,
      sampleCount: field.sampleCount,
      storage: "cpu-dense",
      layout: "x-fastest",
      values,
      signConvention: field.signConvention,
    };
  } finally {
    tracker.destroyAll();
  }
}
