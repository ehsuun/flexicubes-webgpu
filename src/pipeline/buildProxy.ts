import type {
  IndexedMesh,
  WebGpuDenseSdfBakeOptions,
  WebGpuFlexiCubesExtractOptions,
  WebGpuProxyMeshResult,
} from "../core/types.js";
import { extractFlexiCubesWebGpu } from "../flexicubes/webgpu/extract.js";
import { bakeDenseSdfWebGpuResident } from "../sdf/webgpu/bake.js";

export async function buildProxyMeshWebGpu(
  device: GPUDevice,
  mesh: IndexedMesh,
  sdfOptions: WebGpuDenseSdfBakeOptions,
  extractionOptions: WebGpuFlexiCubesExtractOptions = {},
): Promise<WebGpuProxyMeshResult> {
  const startTime = performance.now();
  const sdf = await bakeDenseSdfWebGpuResident(device, mesh, sdfOptions);
  try {
    const extraction = await extractFlexiCubesWebGpu(
      device,
      sdf.field,
      extractionOptions,
    );
    return {
      mesh: extraction.mesh,
      sdfStats: sdf.stats,
      extractionStats: extraction.stats,
      elapsedMs: performance.now() - startTime,
    };
  } finally {
    sdf.field.dispose();
  }
}
