import type {
  GpuDenseScalarField3D,
  IndexedMesh,
  WebGpuDenseSdfBakeOptions,
  WebGpuProxyMeshLodOptions,
  WebGpuProxyMeshLodResult,
  WebGpuProxyMeshLodsResult,
} from "../core/types.js";
import { extractFlexiCubesWebGpu } from "../flexicubes/webgpu/extract.js";
import { bakeDenseSdfWebGpuResident } from "../sdf/webgpu/bake.js";
import { deriveNestedDenseScalarFieldWebGpu } from
  "../sdf/webgpu/deriveNested.js";
import { withAsyncResource } from "./resourceLifetime.js";
import { ProxyOuterEnvelopeError } from "./outerEnvelope.js";
import { extractProxyOuterEnvelope } from "./outerEnvelopeExtraction.js";

type LodOptionsTuple = readonly [
  WebGpuProxyMeshLodOptions,
  WebGpuProxyMeshLodOptions,
  WebGpuProxyMeshLodOptions,
];

const validateLodOptions = (
  fineCellCounts: readonly [number, number, number],
  lodOptions: LodOptionsTuple,
): void => {
  if (lodOptions[0].cellRatio !== 1) {
    throw new RangeError("LOD 0 must use the resident fine field at ratio 1");
  }
  for (let level = 0; level < lodOptions.length; level++) {
    const current = lodOptions[level]!;
    const ratio = current.cellRatio;
    if (!Number.isSafeInteger(ratio) || ratio < 1) {
      throw new RangeError("LOD cell ratios must be positive integers");
    }
    if (
      level > 0
      && ratio <= lodOptions[level - 1]!.cellRatio
    ) {
      throw new RangeError("LOD cell ratios must be strictly increasing");
    }
    if (fineCellCounts.some(count => count % ratio !== 0)) {
      throw new RangeError(
        `fine field cell counts are not divisible by LOD ${level} ratio`,
      );
    }
  }
};

const extractLod = async (
  device: GPUDevice,
  source: IndexedMesh,
  fineField: GpuDenseScalarField3D,
  level: 0 | 1 | 2,
  options: WebGpuProxyMeshLodOptions,
): Promise<WebGpuProxyMeshLodResult> => {
  const derivationStartedAt = performance.now();
  const acquire = (): Promise<GpuDenseScalarField3D> =>
    deriveNestedDenseScalarFieldWebGpu(
      device,
      fineField,
      options.cellRatio,
      {
        ...(options.maxDerivedFieldBytes === undefined
          ? {}
          : { maxGpuBytes: options.maxDerivedFieldBytes }),
        ...(options.extraction?.signal === undefined
          ? {}
          : { signal: options.extraction.signal }),
      },
    );
  const useField = async (
    field: GpuDenseScalarField3D,
  ): Promise<WebGpuProxyMeshLodResult> => {
    const fieldDerivationMs = performance.now() - derivationStartedAt;
    const extractionOptions = options.extraction ?? {};
    let extracted;
    try {
      extracted = options.outerEnvelope === undefined
        ? {
          extraction: await extractFlexiCubesWebGpu(
            device,
            field,
            extractionOptions,
          ),
          evidence: undefined,
        }
        : await extractProxyOuterEnvelope(
          source,
          extractionOptions.isoValue ?? 0,
          options.outerEnvelope,
          extractionOptions.signal,
          isoValue => extractFlexiCubesWebGpu(device, field, {
            ...extractionOptions,
            isoValue,
          }),
        );
    } catch (error) {
      if (error instanceof ProxyOuterEnvelopeError) {
        throw new ProxyOuterEnvelopeError(
          `LOD ${level}: ${error.message}`,
          error.evidence,
        );
      }
      throw error;
    }
    return {
      level,
      cellRatio: options.cellRatio,
      cellCounts: field.cellCounts,
      mesh: extracted.extraction.mesh,
      extractionStats: extracted.extraction.stats,
      ...(extracted.evidence === undefined
        ? {}
        : { outerEnvelope: extracted.evidence }),
      fieldDerivationMs,
    };
  };
  if (options.cellRatio === 1) {
    return useField(fineField);
  }
  return withAsyncResource(acquire, (field) => field.dispose(), useField);
};

const extractLodWithFallback = async (
  device: GPUDevice,
  source: IndexedMesh,
  fineField: GpuDenseScalarField3D,
  level: 0 | 1 | 2,
  options: WebGpuProxyMeshLodOptions,
  previous: WebGpuProxyMeshLodResult | undefined,
): Promise<WebGpuProxyMeshLodResult> => {
  try {
    return await extractLod(device, source, fineField, level, options);
  } catch (error) {
    if (
      !(error instanceof ProxyOuterEnvelopeError)
      || options.outerEnvelopeFallback !== "previous-verified-lod"
      || previous === undefined
    ) {
      throw error;
    }
    const sourceLevel = previous.outerEnvelopeFallback?.sourceLevel
      ?? (previous.level === 0 || previous.level === 1
        ? previous.level
        : undefined);
    if (sourceLevel === undefined) {
      throw new Error("outer-envelope LOD fallback source must be finer");
    }
    return {
      ...previous,
      level,
      outerEnvelopeFallback: {
        kind: "previous-verified-lod",
        sourceLevel,
        rejectedCellRatio: options.cellRatio,
        rejection: error.evidence,
      },
    };
  }
};

export async function buildProxyMeshLodsWebGpu(
  device: GPUDevice,
  mesh: IndexedMesh,
  sdfOptions: WebGpuDenseSdfBakeOptions,
  lodOptions: LodOptionsTuple,
): Promise<WebGpuProxyMeshLodsResult> {
  validateLodOptions(sdfOptions.cellCounts, lodOptions);
  const startTime = performance.now();
  return withAsyncResource(
    () => bakeDenseSdfWebGpuResident(device, mesh, sdfOptions),
    (sdf) => sdf.field.dispose(),
    async (sdf) => {
      const lod0 = await extractLodWithFallback(
        device, mesh, sdf.field, 0, lodOptions[0], undefined,
      );
      const lod1 = await extractLodWithFallback(
        device, mesh, sdf.field, 1, lodOptions[1], lod0,
      );
      const lod2 = await extractLodWithFallback(
        device, mesh, sdf.field, 2, lodOptions[2], lod1,
      );
      return {
        lods: [lod0, lod1, lod2],
        sdfStats: sdf.stats,
        elapsedMs: performance.now() - startTime,
      };
    },
  );
}
