# Architecture

## Objective

Generate a useful proxy mesh from complex indexed geometry in a browser without
blocking the scene, requiring offline DCC work, or coupling the algorithm to a
renderer.

The public system has three independently consumable layers:

```text
MeshInput
   |
   v
SDF builder -----> ScalarField3D
                         |
                         v
                 FlexiCubes extractor
                         |
                         v
                  ExtractedMesh
```

The composed pipeline keeps `ScalarField3D` GPU-resident. Downloading a field
and uploading it again is not an acceptable production path.

## Data contracts

The exact TypeScript names may change before `0.1`, but the semantics may not.

### Indexed mesh

```ts
interface IndexedMesh {
  positions: Float32Array;
  indices: Uint16Array | Uint32Array;
}
```

Positions are tightly packed XYZ coordinates. Input validation, transforms, and
multi-primitive merging are pure helpers. Renderer mesh objects are never part
of the core API.

### Scalar field

```ts
interface ScalarField3D {
  values: Float32Array;
  sampleCounts: readonly [number, number, number];
  sampleOrigin: readonly [number, number, number];
  sampleSpacing: readonly [number, number, number];
  signConvention: "negative-inside";
}
```

Values live at **lattice vertices**, not cell centers. A grid containing
`[nx, ny, nz]` cells therefore contains
`[(nx + 1), (ny + 1), (nz + 1)]` samples.

This is a deliberate compatibility boundary. FlexiCubes classifies each cell
from its eight corner samples. Treating a cell-centered volume as vertex
samples silently moves the represented domain by half a voxel and loses the
outer sample plane.

GPU-resident fields use the same metadata plus an owned `GPUBuffer`. Ownership
and disposal are explicit. A download method is optional and asynchronous.

### Extracted mesh

```ts
interface ExtractedMesh {
  positions: Float32Array;
  indices: Uint32Array;
  sourceCells?: Uint32Array;
}
```

The extractor returns geometry and enough provenance for a consumer to transfer
attributes. Materials, UV projection, mesh partitioning, Babylon vertex
buffers, and surface-cache card generation belong to the consumer.

## SDF construction

### Sign is a policy

Nearest-triangle normals do not define a reliable sign for concave,
self-intersecting, open, or inconsistently wound meshes. The SDF API must make
its geometry assumption explicit:

- `closed`: robust inside/outside classification for a closed, consistently
  wound surface.
- `parity`: multi-direction ray parity with deterministic handling of shared
  edges and vertices.
- `shell`: unsigned distance minus a requested thickness for open or thin
  geometry such as walls and cards.
- `unsigned`: distance only, for consumers that do not extract a closed
  isosurface.

The first production proxy path should implement `parity` and `shell`.
Generalized winding is a later quality option, not a prerequisite for bounded
runtime behavior.

### Acceleration

The current target is a GPU-built spatial acceleration structure. A CPU
reference may use straightforward bins or a BVH, but the browser path must not
create a JavaScript array for every spatial bin and synchronously distribute
millions of triangles across those arrays.

Candidate production sequence:

1. Upload indexed positions and indices.
2. Compute transformed triangle bounds and normals.
3. Count triangle references per coarse bin.
4. Prefix-sum counts and scatter triangle references.
5. Evaluate unsigned distance for field samples in bounded tiles.
6. Classify sign according to the selected policy.
7. Emit either a dense field or compact active bricks.

Every dispatch is bounded by device limits. Large jobs expose progress and
cancellation between tiles.

### Dense and sparse fields

Dense fields are the correctness baseline and are appropriate at modest
resolution. Sparse bricks reduce storage and extraction work when occupancy is
low, but they are an optimization, not a separate semantic format.

Sparse bricks must include a coarse fallback field or explicit outside value so
distance queries remain defined away from active bricks. Brick activation must
include a conservative halo large enough for zero-crossing extraction.

## FlexiCubes extraction

The implementation is staged:

1. A deterministic CPU port of fixed-field triangular extraction.
2. Golden comparisons with the Apache-2.0 PyTorch reference.
3. WebGPU classification, prefix-sum allocation, dual-vertex computation, and
   triangulation.
4. Optional gradient/QEF path for sharper fixed-field output.

The first release uses fixed fields and uniform FlexiCubes weights. It does not
claim the optimization-time quality improvements from the paper, which rely on
learned or optimized flexible parameters.

Lookup tables adapted from the official implementation retain upstream notices
and modification markers.

## WebGPU boundary

The primary backend accepts standards-based WebGPU objects, beginning with a
`GPUDevice`. It does not import a renderer.

Two API levels are planned:

- High level: typed arrays in, typed arrays out.
- Composed GPU: existing GPU buffers in, GPU buffers out, with explicit
  layouts and ownership.

Renderer adapters may live in optional packages, but cannot become dependencies
of the core. If an engine does not expose its `GPUDevice` as a stable public
contract, that engine adapter should execute the shared WGSL/stage plan through
the engine's supported compute API instead of reaching into private fields.

## Scheduling and failure contract

Runtime proxy generation is opportunistic background work. The library must
never express "eventually finishes" as its only contract.

Every production job accepts:

- maximum output resolution/cells
- maximum GPU memory
- maximum output triangles
- per-submit work budget
- `AbortSignal`
- progress callback

Planning returns an estimate before heavy work begins. If the requested job
cannot fit the declared budget, the library returns a typed refusal or a
smaller proposed plan. It does not start an unbounded fallback automatically.

Application rendering must continue while a job is pending. The last valid
proxy, direct simple geometry, or a conservative bounding proxy remains active
until replacement succeeds.

## Correctness and performance gates

No backend is production-ready until it passes:

- analytic field fixtures and empty/full edge cases
- topology-table comparisons with the reference implementation
- CPU/WebGPU output agreement within documented tolerances
- sign tests for closed, open, thin, and reversed-winding fixtures
- cancellation and device-loss tests
- peak-memory accounting
- main-thread long-task measurement
- browser benchmarks at 10K, 100K, 1M, and multi-million input triangles

The primary latency metrics are:

- time until control returns to the application
- longest main-thread task
- total proxy-ready time
- peak CPU and GPU memory

Total completion time alone cannot certify an interactive path.

## Product boundary

The library owns geometry processing. A consuming application owns:

- deciding whether simple geometry bypasses proxy generation
- selecting quality and time budgets
- cache invalidation and persistence
- scene lifecycle and fallback rendering
- materials and attribute transfer
- lighting/GI policy

This separation lets Vutify use the same pipeline for runtime GI proxies and
offline-quality baking without turning the public library into a GI system.
