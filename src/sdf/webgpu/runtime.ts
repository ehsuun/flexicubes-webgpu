import type {
  IndexedMesh,
  SdfBakePhase,
  WebGpuDenseSdfBakeOptions,
} from "../../core/types.js";
import { SdfBakeAbortError } from "../cpuReference.js";
import { SdfGpuBudgetError } from "./planner.js";

const COPY_CHUNK_ELEMENTS = 65_536;

export function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw new SdfBakeAbortError();
  }
}

function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

export function reportProgress(
  options: WebGpuDenseSdfBakeOptions,
  phase: SdfBakePhase,
  completed: number,
  total: number,
): void {
  options.onProgress?.({
    phase,
    completed,
    total,
    fraction: total === 0 ? 1 : completed / total,
  });
}

export interface PreparedGpuMesh {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
}

export async function prepareMeshForWebGpu(
  mesh: IndexedMesh,
  options: WebGpuDenseSdfBakeOptions,
): Promise<PreparedGpuMesh> {
  const totalElements = mesh.positions.length + mesh.indices.length;
  let completed = 0;
  reportProgress(options, "prepare", completed, totalElements);

  for (
    let start = 0;
    start < mesh.positions.length;
    start += COPY_CHUNK_ELEMENTS
  ) {
    throwIfAborted(options.signal);
    const end = Math.min(start + COPY_CHUNK_ELEMENTS, mesh.positions.length);
    for (let index = start; index < end; index++) {
      if (!Number.isFinite(mesh.positions[index])) {
        throw new RangeError(`positions[${index}] must be finite`);
      }
    }
    completed += end - start;
    reportProgress(options, "prepare", completed, totalElements);
    await yieldToEventLoop();
  }

  const vertexCount = mesh.positions.length / 3;
  const indices = mesh.indices instanceof Uint32Array
    ? mesh.indices
    : new Uint32Array(mesh.indices.length);
  for (
    let start = 0;
    start < mesh.indices.length;
    start += COPY_CHUNK_ELEMENTS
  ) {
    throwIfAborted(options.signal);
    const end = Math.min(start + COPY_CHUNK_ELEMENTS, mesh.indices.length);
    for (let index = start; index < end; index++) {
      const vertexIndex = mesh.indices[index]!;
      if (vertexIndex >= vertexCount) {
        throw new RangeError(`indices[${index}] references a missing vertex`);
      }
      if (indices !== mesh.indices) {
        indices[index] = vertexIndex;
      }
    }
    completed += end - start;
    reportProgress(options, "prepare", completed, totalElements);
    await yieldToEventLoop();
  }

  return {
    positions: mesh.positions,
    indices,
  };
}

interface TrackedBuffer {
  readonly buffer: GPUBuffer;
  readonly bytes: number;
}

export class GpuBufferTracker {
  readonly #device: GPUDevice;
  readonly #maxBytes: number;
  readonly #budgetError: (message: string) => Error;
  readonly #buffers = new Map<GPUBuffer, TrackedBuffer>();
  #currentBytes = 0;
  #peakBytes = 0;

  public constructor(
    device: GPUDevice,
    maxBytes: number,
    budgetError: (message: string) => Error = (
      message,
    ) => new SdfGpuBudgetError(message),
  ) {
    this.#device = device;
    this.#maxBytes = maxBytes;
    this.#budgetError = budgetError;
  }

  public get peakBytes(): number {
    return this.#peakBytes;
  }

  public create(
    label: string,
    requestedBytes: number,
    usage: GPUBufferUsageFlags,
    storageBinding: boolean,
  ): GPUBuffer {
    const bytes = Math.max(4, Math.ceil(requestedBytes / 4) * 4);
    if (bytes > this.#device.limits.maxBufferSize) {
      throw this.#budgetError(
        `${label} exceeds maxBufferSize (${bytes} bytes)`,
      );
    }
    if (
      storageBinding
      && bytes > this.#device.limits.maxStorageBufferBindingSize
    ) {
      throw this.#budgetError(
        `${label} exceeds maxStorageBufferBindingSize (${bytes} bytes)`,
      );
    }
    if (this.#currentBytes + bytes > this.#maxBytes) {
      throw this.#budgetError(
        `${label} would exceed the ${this.#maxBytes}-byte GPU budget`,
      );
    }

    const buffer = this.#device.createBuffer({ label, size: bytes, usage });
    this.#buffers.set(buffer, { buffer, bytes });
    this.#currentBytes += bytes;
    this.#peakBytes = Math.max(this.#peakBytes, this.#currentBytes);
    return buffer;
  }

  public destroy(buffer: GPUBuffer): void {
    const tracked = this.#buffers.get(buffer);
    if (tracked === undefined) {
      return;
    }
    buffer.destroy();
    this.#buffers.delete(buffer);
    this.#currentBytes -= tracked.bytes;
  }

  public release(buffer: GPUBuffer): void {
    const tracked = this.#buffers.get(buffer);
    if (tracked === undefined) {
      throw new Error("cannot release an untracked GPU buffer");
    }
    this.#buffers.delete(buffer);
    this.#currentBytes -= tracked.bytes;
  }

  public destroyAll(): void {
    for (const tracked of this.#buffers.values()) {
      tracked.buffer.destroy();
    }
    this.#buffers.clear();
    this.#currentBytes = 0;
  }
}

export async function writeBufferChunked(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: Float32Array | Uint32Array,
  options: WebGpuDenseSdfBakeOptions,
  progressOffset: number,
  progressTotal: number,
): Promise<number> {
  let completed = progressOffset;
  for (let start = 0; start < data.length; start += COPY_CHUNK_ELEMENTS) {
    throwIfAborted(options.signal);
    const end = Math.min(start + COPY_CHUNK_ELEMENTS, data.length);
    const chunk = data.subarray(start, end);
    device.queue.writeBuffer(
      buffer,
      start * data.BYTES_PER_ELEMENT,
      chunk,
    );
    completed += end - start;
    reportProgress(options, "upload", completed, progressTotal);
    await yieldToEventLoop();
  }
  return completed;
}

export async function readBufferCopy(
  device: GPUDevice,
  tracker: GpuBufferTracker,
  source: GPUBuffer,
  bytes: number,
  label: string,
  signal: AbortSignal | undefined,
): Promise<ArrayBuffer> {
  throwIfAborted(signal);
  const staging = tracker.create(
    `${label} readback`,
    bytes,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    false,
  );
  const encoder = device.createCommandEncoder({ label: `${label} copy` });
  encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  throwIfAborted(signal);
  const copy = staging.getMappedRange(0, bytes).slice(0);
  staging.unmap();
  tracker.destroy(staging);
  return copy;
}
