import type {
  IndexedMesh,
  WebGpuDenseSdfBakeOptions,
  WebGpuFlexiCubesExtractOptions,
  ProxyOuterEnvelopeOptions,
  WebGpuProxyMeshResult,
} from "../core/types.js";
import { extractFlexiCubesWebGpu } from "../flexicubes/webgpu/extract.js";
import { bakeDenseSdfWebGpuResident } from "../sdf/webgpu/bake.js";
import { extractProxyOuterEnvelope } from "./outerEnvelopeExtraction.js";

export async function buildProxyMeshWebGpu(
  device: GPUDevice,
  mesh: IndexedMesh,
  sdfOptions: WebGpuDenseSdfBakeOptions,
  extractionOptions: WebGpuFlexiCubesExtractOptions = {},
  outerEnvelopeOptions?: ProxyOuterEnvelopeOptions,
): Promise<WebGpuProxyMeshResult> {
  const startTime = performance.now();
  const sdf = await bakeDenseSdfWebGpuResident(device, mesh, sdfOptions);
  try {
    const extracted = outerEnvelopeOptions === undefined
      ? {
        extraction: await extractFlexiCubesWebGpu(
          device,
          sdf.field,
          extractionOptions,
        ),
        evidence: undefined,
      }
      : await extractProxyOuterEnvelope(
        mesh,
        extractionOptions.isoValue ?? 0,
        outerEnvelopeOptions,
        extractionOptions.signal,
        isoValue => extractFlexiCubesWebGpu(device, sdf.field, {
          ...extractionOptions,
          isoValue,
        }),
      );
    return {
      mesh: extracted.extraction.mesh,
      sdfStats: sdf.stats,
      extractionStats: extracted.extraction.stats,
      ...(extracted.evidence === undefined
        ? {}
        : { outerEnvelope: extracted.evidence }),
      elapsedMs: performance.now() - startTime,
    };
  } finally {
    sdf.field.dispose();
  }
}
