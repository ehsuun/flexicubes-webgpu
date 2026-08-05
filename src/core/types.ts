export type Vec3 = readonly [x: number, y: number, z: number];
export type GridSize3 = readonly [x: number, y: number, z: number];

export interface Bounds3 {
  readonly min: Vec3;
  readonly max: Vec3;
}

export interface IndexedMesh {
  readonly positions: Float32Array;
  readonly indices: Uint16Array | Uint32Array;
}

export type Matrix4 =
  | Float32Array
  | Float64Array
  | readonly number[];

export interface Lattice3D {
  readonly cellCounts: GridSize3;
  readonly sampleCounts: GridSize3;
  readonly sampleOrigin: Vec3;
  readonly sampleSpacing: Vec3;
  readonly sampleCount: number;
}

export type SdfSignPolicy =
  | { readonly kind: "parity" }
  | {
    readonly kind: "parity-shell-union";
    readonly halfThickness: number;
  }
  | { readonly kind: "shell"; readonly halfThickness: number }
  | { readonly kind: "unsigned" };

export type ScalarFieldSignConvention = "negative-inside" | "unsigned";

export interface DenseScalarField3D extends Lattice3D {
  readonly storage: "cpu-dense";
  readonly layout: "x-fastest";
  readonly values: Float32Array;
  readonly signConvention: ScalarFieldSignConvention;
}

export interface GpuDenseScalarField3D extends Lattice3D {
  readonly storage: "gpu-dense";
  readonly layout: "x-fastest";
  readonly device: GPUDevice;
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly signConvention: ScalarFieldSignConvention;
  readonly disposed: boolean;
  dispose(): void;
}

export type SdfBakePhase =
  | "prepare"
  | "upload"
  | "bin-count"
  | "bin-scatter"
  | "distance-and-sign"
  | "readback";

export interface SdfBakeProgress {
  readonly phase: SdfBakePhase;
  readonly completed: number;
  readonly total: number;
  readonly fraction: number;
}

export interface DenseSdfBakeOptions {
  readonly domain: Bounds3;
  readonly cellCounts: GridSize3;
  readonly signPolicy: SdfSignPolicy;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: SdfBakeProgress) => void;
}

export interface CpuSdfBakeStats {
  readonly backend: "cpu-reference";
  readonly triangleCount: number;
  readonly sampleCount: number;
  readonly memoryBytes: number;
  readonly distanceTests: number;
  readonly parityRayTriangleTests: number;
  readonly elapsedMs: number;
}

export interface CpuSdfBakeResult {
  readonly field: DenseScalarField3D;
  readonly stats: CpuSdfBakeStats;
}

export interface WebGpuSdfExecutionLimits {
  readonly maxGpuBytes?: number;
  readonly maxTriangleReferences?: number;
  readonly triangleBatchSize?: number;
  readonly sampleBatchSize?: number;
  readonly distanceBinResolution?: number;
  readonly parityBinResolution?: number;
}

export interface WebGpuDenseSdfBakeOptions extends DenseSdfBakeOptions {
  readonly execution?: WebGpuSdfExecutionLimits;
}

export interface WebGpuSdfBakeStats {
  readonly backend: "webgpu-dense-binned";
  readonly triangleCount: number;
  readonly sampleCount: number;
  readonly memoryBytes: number;
  readonly distanceBinResolution: number;
  readonly parityBinResolution: number;
  readonly distanceTriangleReferences: number;
  readonly parityTriangleReferences: number;
  readonly elapsedMs: number;
}

export interface WebGpuSdfBakeResult {
  readonly field: DenseScalarField3D;
  readonly stats: WebGpuSdfBakeStats;
}

export interface WebGpuResidentSdfBakeResult {
  readonly field: GpuDenseScalarField3D;
  readonly stats: WebGpuSdfBakeStats;
}

export type FlexiCubesPhase =
  | "classify-cells"
  | "resolve-cases"
  | "build-surface-edges"
  | "build-dual-vertices"
  | "triangulate"
  | "readback";

export interface FlexiCubesProgress {
  readonly phase: FlexiCubesPhase;
  readonly completed: number;
  readonly total: number;
  readonly fraction: number;
}

export interface FlexiCubesExtractOptions {
  readonly isoValue?: number;
  readonly maxOutputTriangles?: number;
  readonly allowBoundaryOpen?: boolean;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: FlexiCubesProgress) => void;
}

export interface ExtractedMesh {
  readonly positions: Float32Array;
  readonly indices: Uint32Array;
  readonly sourceCells: Uint32Array;
}

export interface FlexiCubesExtractionStats {
  readonly backend: "cpu-reference";
  readonly surfaceCellCount: number;
  readonly surfaceEdgeCount: number;
  readonly boundarySurfaceEdgeCount: number;
  readonly dualVertexCount: number;
  readonly quadCount: number;
  readonly triangleCount: number;
  readonly elapsedMs: number;
}

export interface FlexiCubesExtractionResult {
  readonly mesh: ExtractedMesh;
  readonly stats: FlexiCubesExtractionStats;
}

export interface WebGpuFlexiCubesExecutionLimits {
  readonly maxGpuBytes?: number;
  readonly cellBatchSize?: number;
  readonly edgeBatchSize?: number;
}

export interface WebGpuFlexiCubesExtractOptions
extends FlexiCubesExtractOptions {
  readonly execution?: WebGpuFlexiCubesExecutionLimits;
}

export interface WebGpuFlexiCubesExtractionStats {
  readonly backend: "webgpu";
  readonly surfaceCellCount: number;
  readonly surfaceEdgeCount: number;
  readonly boundarySurfaceEdgeCount: number;
  readonly dualVertexCount: number;
  readonly quadCount: number;
  readonly triangleCount: number;
  readonly memoryBytes: number;
  readonly elapsedMs: number;
}

export interface WebGpuFlexiCubesExtractionResult {
  readonly mesh: ExtractedMesh;
  readonly stats: WebGpuFlexiCubesExtractionStats;
}

export interface WebGpuProxyMeshResult {
  readonly mesh: ExtractedMesh;
  readonly sdfStats: WebGpuSdfBakeStats;
  readonly extractionStats: WebGpuFlexiCubesExtractionStats;
  readonly outerEnvelope?: ProxyOuterEnvelopeEvidence;
  readonly elapsedMs: number;
}

export interface ProxyOuterEnvelopeProgress {
  readonly attempt: number;
  readonly maximumAttempts: number;
  readonly isoValue: number;
  readonly completedSamples: number;
  readonly totalSamples: number;
}

export interface ProxyOuterEnvelopeVerificationOptions {
  readonly minimumSeparation: number;
  readonly maximumSourceSamples?: number;
  readonly sampleBatchSize?: number;
  readonly signal?: AbortSignal;
}

export interface ProxyOuterEnvelopeOptions {
  readonly minimumSeparation: number;
  readonly maximumExpansion: number;
  readonly maximumAttempts?: number;
  readonly maximumSourceSamples?: number;
  readonly sampleBatchSize?: number;
  readonly onProgress?: (progress: ProxyOuterEnvelopeProgress) => void;
}

export interface ProxyOuterEnvelopeVerificationEvidence {
  readonly method: "sampled-source-surface";
  readonly sampleStrategy: "uniform-vertices-and-triangle-centroids";
  readonly sourceSampleCount: number;
  readonly violationCount: number;
  readonly minimumSignedSeparation: number;
  readonly maximumIngress: number;
  readonly queryCount: number;
}

export interface ProxyOuterEnvelopeEvidence {
  readonly attempts: number;
  readonly initialIsoValue: number;
  readonly finalIsoValue: number;
  readonly initialVerification: ProxyOuterEnvelopeVerificationEvidence;
  readonly finalVerification: ProxyOuterEnvelopeVerificationEvidence;
  readonly elapsedMs: number;
}

export interface WebGpuProxyMeshLodOptions {
  readonly cellRatio: number;
  readonly extraction?: WebGpuFlexiCubesExtractOptions;
  readonly outerEnvelope?: ProxyOuterEnvelopeOptions;
  readonly outerEnvelopeFallback?: "previous-verified-lod";
  readonly maxDerivedFieldBytes?: number;
}

export interface ProxyOuterEnvelopeLodFallbackEvidence {
  readonly kind: "previous-verified-lod";
  readonly sourceLevel: 0 | 1;
  readonly rejectedCellRatio: number;
  readonly rejection: ProxyOuterEnvelopeEvidence;
}

export interface WebGpuProxyMeshLodResult {
  readonly level: 0 | 1 | 2;
  readonly cellRatio: number;
  readonly cellCounts: GridSize3;
  readonly mesh: ExtractedMesh;
  readonly extractionStats: WebGpuFlexiCubesExtractionStats;
  readonly outerEnvelope?: ProxyOuterEnvelopeEvidence;
  readonly outerEnvelopeFallback?: ProxyOuterEnvelopeLodFallbackEvidence;
  readonly fieldDerivationMs: number;
}

export interface WebGpuProxyMeshLodsResult {
  readonly lods: readonly [
    WebGpuProxyMeshLodResult,
    WebGpuProxyMeshLodResult,
    WebGpuProxyMeshLodResult,
  ];
  readonly sdfStats: WebGpuSdfBakeStats;
  readonly elapsedMs: number;
}
