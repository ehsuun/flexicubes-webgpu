# flexicubes-webgpu

Browser-native mesh-to-SDF and FlexiCubes isosurface extraction for TypeScript
and WebGPU.

> **Status:** experimental alpha. The CPU references and dense WebGPU pipeline
> are implemented and tested, but the API may change before `0.1.0`. Consume an
> immutable Git commit during early integration. No npm release exists yet.

## Why this exists

Browser 3D tools should be able to create bounded collision, occlusion, and
lighting proxies at runtime. Requiring every user to preprocess a scene in a
DCC application defeats that goal.

This project provides a renderer-neutral pipeline:

```text
indexed triangle mesh -> signed scalar field -> bounded proxy mesh
```

Each stage is also usable independently:

- **SDF:** build a dense signed field from indexed geometry.
- **FlexiCubes:** extract a mesh from an existing scalar field.
- **Pipeline:** keep the field on the GPU and produce a proxy under explicit
  memory, reference-count, batch-size, and triangle-count limits.

The first production consumer is expected to be Vutify, a browser-based music
visualization application where the proxy supports runtime surface-cache
lighting. The library itself will not contain Vutify, Babylon.js, GI, material,
or scene-lifecycle code.

## Implemented guarantees

- Renderer-neutral TypeScript contracts and raw WebGPU compute.
- No work proportional to millions of input triangles runs synchronously on
  the browser UI thread.
- Explicit memory and work budgets, cancellation, and progress reporting.
- GPU-resident composition between SDF construction and mesh extraction; CPU
  readback is optional.
- Deterministic CPU references for correctness testing.
- Clear behavior for closed, open, thin, and inconsistently wound geometry.
- Measured fallbacks instead of an unbounded "enable and wait" path.

The dense backend builds distance and parity bins on the GPU. JavaScript
validates and uploads input in bounded asynchronous chunks, so preparation does
not synchronously walk millions of triangles before returning control. The
composed path reads back only small allocation counters and the final compact
mesh; it does not download the dense scalar field.

See [Architecture](docs/ARCHITECTURE.md) for contracts and limitations and
[SDF migration](docs/SDF_MIGRATION.md) for implementation status.

## Quick start

```ts
import { buildProxyMeshWebGpu } from "flexicubes-webgpu/pipeline";

const adapter = await navigator.gpu.requestAdapter({
  powerPreference: "high-performance",
});
if (adapter === null) {
  throw new Error("WebGPU is unavailable");
}
const device = await adapter.requestDevice();

const result = await buildProxyMeshWebGpu(
  device,
  { positions, indices },
  {
    domain: {
      min: [-1.1, -1.1, -1.1],
      max: [1.1, 1.1, 1.1],
    },
    cellCounts: [48, 48, 48],
    signPolicy: { kind: "parity" },
    execution: {
      maxGpuBytes: 256 * 1024 * 1024,
      maxTriangleReferences: 12_000_000,
      triangleBatchSize: 65_536,
      sampleBatchSize: 4096,
    },
  },
  {
    maxOutputTriangles: 200_000,
    execution: {
      maxGpuBytes: 256 * 1024 * 1024,
    },
  },
);

// result.mesh.positions, result.mesh.indices, result.mesh.sourceCells
```

Use `bakeDenseSdfWebGpuResident` and `extractFlexiCubesWebGpu` separately when
the application needs to retain or reuse a field. A resident field is owned,
belongs to the `GPUDevice` that created it, and must be disposed.

Call `prewarmSdfWebGpu(device)` and `prewarmFlexiCubesWebGpu(device)` during an
idle/loading phase when first-use shader compilation latency matters.

### Multi-LOD proxy generation

`buildProxyMeshLodsWebGpu` bakes one fine SDF and extracts three aligned proxy
meshes without downloading or rebaking the field:

```ts
import { buildProxyMeshLodsWebGpu } from "flexicubes-webgpu/pipeline";

const result = await buildProxyMeshLodsWebGpu(
  device,
  { positions, indices },
  {
    domain,
    cellCounts: [24, 24, 24],
    signPolicy: { kind: "parity" },
  },
  [
    { cellRatio: 1 },
    { cellRatio: 2 },
    { cellRatio: 4 },
  ],
);
```

The result contains LODs at 24³, 12³, and 6³ cells. Coarse values come from
coincident fine samples, not signed-distance averaging. Every temporary GPU
field is owned by the call and released after extraction, including failure
and cancellation paths.

## Geometry policy

- Use `parity` for closed geometry. It uses three jittered axis rays and a
  majority vote, so reversed input winding is supported.
- Use `parity-shell-union` for mostly closed geometry whose open, thin, or
  sparsely defective features must survive a coarse field. It preserves the
  parity interior and unions it with an explicit two-sided rescue shell.
- Use `shell` with an explicit half-thickness for open surfaces, cards, and
  thin walls.
- Use `unsigned` only when no zero-isosurface extraction is required.

Parity is not a general repair operation for self-intersecting or non-manifold
input. `parity-shell-union` is likewise not automatic topology repair: its
thickness is a product policy that deliberately expands every source surface.
Applications should keep their previous proxy or a conservative fallback when
a job refuses, is cancelled, or receives unsuitable geometry.

## Scope

This alpha targets dense fixed-field extraction for practical runtime proxy
generation. Sparse fields, automatic differentiation, training-time
regularizers, tetrahedral output, adaptive grids, renderer materials, UV
transfer, and GI policy are not implemented.

FlexiCubes improves the mesh extracted from a field; it does not make
mesh-to-field conversion cheap by itself. The SDF builder and its bounded GPU
acceleration are therefore first-class parts of this project.

## Development and releases

This repository is the canonical source. Applications should consume immutable
commits during early integration and exact `0.x` package versions after the
first prerelease. Application repositories should keep only their adapter,
scheduling policy, cache ownership, and product-specific fallback logic.

The repository is currently consumed from immutable Git revisions. An npm
package should be published only after a broader browser/device matrix and
device-loss coverage are complete.

## Verification

```sh
npm ci
npm run check
npm run test:webgpu
npm run benchmark:webgpu
```

`npm run test:webgpu` compares CPU and GPU fields, exercises closed and open
geometry policies, verifies CPU/GPU FlexiCubes agreement, and checks that the
GPU result is a closed outward-wound two-manifold.

The stress command builds a signed 24-cell field from one million generated
triangles while measuring initial call return and main-thread timer gaps.
Input generation is intentionally outside the measured library interval.

On 2026-07-25, headless Chrome on an NVIDIA Ampere adapter completed that
synthetic signed workload in about 2.14 seconds, returned from the initial call
in about 1.3 milliseconds, observed a maximum timer gap of about 6.7
milliseconds, and reported about 64.6 MB peak GPU allocation. This is one
machine and one synthetic workload, not a cross-device performance guarantee.

## FlexiCubes attribution

This is an independent TypeScript/WebGPU implementation and is not an official
NVIDIA project. It is based on the FlexiCubes method and informed by NVIDIA's
Apache-2.0 reference implementation:

- [Project and paper](https://research.nvidia.com/labs/toronto-ai/flexicubes/)
- [Official PyTorch implementation](https://github.com/nv-tlabs/FlexiCubes)

Files adapted from the reference implementation will retain its copyright and
license notices and will state that they were modified. See [NOTICE](NOTICE).

If this project helps your research, cite the original work:

```bibtex
@article{shen2023flexicubes,
  author = {Shen, Tianchang and Munkberg, Jacob and Hasselgren, Jon and
            Yin, Kangxue and Wang, Zian and Chen, Wenzheng and Gojcic, Zan and
            Fidler, Sanja and Sharp, Nicholas and Gao, Jun},
  title = {Flexible Isosurface Extraction for Gradient-Based Mesh Optimization},
  journal = {ACM Transactions on Graphics},
  volume = {42},
  number = {4},
  year = {2023},
  doi = {10.1145/3592430}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
