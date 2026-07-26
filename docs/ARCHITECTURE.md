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

The TypeScript names may change during alpha, but the semantics below are
treated as compatibility boundaries.

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
interface DenseScalarField3D {
  storage: "cpu-dense";
  values: Float32Array;
  cellCounts: readonly [number, number, number];
  sampleCounts: readonly [number, number, number];
  sampleOrigin: readonly [number, number, number];
  sampleSpacing: readonly [number, number, number];
  signConvention: "negative-inside" | "unsigned";
}
```

Values live at **lattice vertices**, not cell centers. A grid containing
`[nx, ny, nz]` cells therefore contains
`[(nx + 1), (ny + 1), (nz + 1)]` samples.

This is a deliberate compatibility boundary. FlexiCubes classifies each cell
from its eight corner samples. Treating a cell-centered volume as vertex
samples silently moves the represented domain by half a voxel and loses the
outer sample plane.

GPU-resident fields use the same metadata plus their creating `GPUDevice` and
an owned `GPUBuffer`. Ownership and disposal are explicit. A field cannot be
used with a different device. Download is optional and asynchronous.

### Extracted mesh

```ts
interface ExtractedMesh {
  positions: Float32Array;
  indices: Uint32Array;
  sourceCells: Uint32Array;
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

- `parity`: multi-direction ray parity with deterministic handling of shared
  edges and vertices.
- `shell`: unsigned distance minus a requested thickness for open or thin
  geometry such as walls and cards.
- `unsigned`: distance only, for consumers that do not extract a closed
  isosurface.

The dense backend implements `parity`, `shell`, and `unsigned`. Generalized
winding is a later quality option, not a prerequisite for bounded runtime
behavior.

### Acceleration

The current backend builds its spatial acceleration structure on the GPU. The
CPU reference is deliberately straightforward, but the browser path does not
create a JavaScript array for every spatial bin or synchronously distribute
millions of triangles across those arrays.

Implemented dense sequence:

1. Upload indexed positions and indices.
2. Compute transformed triangle bounds and normals.
3. Count triangle references per coarse distance and projected-parity bin.
4. Read back the small count buffers, prefix them on the CPU, and scatter
   triangle references on the GPU.
5. Evaluate unsigned distance for field samples in bounded tiles.
6. Classify sign according to the selected policy.
7. Emit a dense GPU-resident field.

Every dispatch is bounded by device limits. Large jobs expose progress and
cancellation between batches. Prefixing coarse count buffers is small and
independent of input triangle count; a future scan kernel can remove that
readback if much larger bin grids become useful.

### Dense and sparse fields

Dense fields are the correctness baseline and are appropriate at modest
resolution. Sparse bricks can reduce storage and extraction work when
occupancy is low, but they are a future optimization, not a separate semantic
format.

Sparse bricks must include a coarse fallback field or explicit outside value so
distance queries remain defined away from active bricks. Brick activation must
include a conservative halo large enough for zero-crossing extraction.

## FlexiCubes extraction

The implemented fixed-field path contains:

1. A deterministic CPU port of fixed-field triangular extraction.
2. Lookup tables generated from the pinned Apache-2.0 PyTorch reference.
3. WebGPU classification, atomic compact allocation, dual-vertex computation,
   and triangulation.
4. CPU/GPU topology, position, manifold, and orientation checks.

The alpha uses fixed fields and uniform FlexiCubes weights. It does not
claim the optimization-time quality improvements from the paper, which rely on
learned or optimized flexible parameters.

Lookup tables adapted from the official implementation retain upstream notices
and modification markers.

## WebGPU boundary

The primary backend accepts standards-based WebGPU objects, beginning with a
`GPUDevice`. It does not import a renderer.

Two API levels are implemented:

- High level: typed arrays in, typed arrays out.
- Composed GPU: typed arrays in, a GPU-resident scalar field between stages,
  and a compact typed-array mesh out.

Accepting externally owned position/index buffers and returning GPU-owned mesh
buffers are future API levels, not current claims.

Renderer adapters may live in optional packages, but cannot become dependencies
of the core. If an engine does not expose its `GPUDevice` as a stable public
contract, that engine adapter should execute the shared WGSL/stage plan through
the engine's supported compute API instead of reaching into private fields.

## Scheduling and failure contract

Runtime proxy generation is opportunistic background work. The library must
never express "eventually finishes" as its only contract.

Production entry points accept:

- explicit output resolution/cells
- maximum GPU memory
- maximum output triangles
- per-submit work budget
- `AbortSignal`
- progress callback

Planning returns an estimate before heavy work begins. If the requested job
cannot fit the declared budget, the library throws a typed budget error before
the refused allocation. It does not choose a smaller plan or start an
unbounded fallback automatically.

Application rendering must continue while a job is pending. The last valid
proxy, direct simple geometry, or a conservative bounding proxy remains active
until replacement succeeds.

## Correctness and performance gates

Current automated gates cover:

- analytic field fixtures and empty/full edge cases
- all 256 topology-table cases and ambiguity inversions from the reference
- CPU/WebGPU output agreement within documented tolerances
- sign tests for closed, open, thin, and reversed-winding fixtures
- cancellation tests
- peak-memory accounting
- main-thread long-task measurement
- a one-million-triangle browser benchmark

Before a stable release, coverage still needs device loss, multiple browser and
adapter families, 10K/100K/multi-million benchmark points, and sparse-field
boundary agreement.

The primary latency metrics are:

- time until control returns to the application
- longest main-thread task
- total proxy-ready time
- peak CPU and GPU memory

Total completion time alone cannot certify an interactive path.

## Known limitations

- Dense storage scales with lattice volume and is intentionally bounded.
- Three-axis parity is robust to winding reversal but cannot make arbitrary
  self-intersecting or non-manifold input well-defined.
- Open and thin geometry needs explicit shell thickness.
- Cancellation occurs between asynchronous upload/dispatch batches; submitted
  GPU work itself cannot be recalled.
- Final positions, indices, and provenance are read back to CPU. Only the
  intermediate scalar field stays resident.
- Uniform fixed-field weights do not provide the paper's
  optimization/training-time benefits.

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
