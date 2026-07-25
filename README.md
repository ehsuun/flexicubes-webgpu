# flexicubes-webgpu

Browser-native mesh-to-SDF and FlexiCubes isosurface extraction for TypeScript
and WebGPU.

> **Status:** pre-alpha. The repository and architecture are public so the
> implementation can be developed in the open, but there is no stable package
> or API release yet.

## Why this exists

Browser 3D tools should be able to create bounded collision, occlusion, and
lighting proxies at runtime. Requiring every user to preprocess a scene in a
DCC application defeats that goal.

This project is intended to provide a renderer-neutral pipeline:

```text
indexed triangle mesh -> signed scalar field -> bounded proxy mesh
```

Each stage will also be usable independently:

- **SDF:** build a dense or sparse signed field from indexed geometry.
- **FlexiCubes:** extract a mesh from an existing scalar field.
- **Pipeline:** keep intermediate data on the GPU and produce a proxy under an
  explicit time, memory, and triangle budget.

The first production consumer is expected to be Vutify, a browser-based music
visualization application where the proxy supports runtime surface-cache
lighting. The library itself will not contain Vutify, Babylon.js, GI, material,
or scene-lifecycle code.

## Design promises

- Renderer-neutral TypeScript contracts and raw WebGPU compute.
- No work proportional to millions of input triangles runs synchronously on
  the browser UI thread.
- Explicit memory and work budgets, cancellation, and progress reporting.
- GPU-resident composition between SDF construction and mesh extraction; CPU
  readback is optional.
- Deterministic CPU references for correctness testing.
- Clear behavior for closed, open, thin, and inconsistently wound geometry.
- Measured fallbacks instead of an unbounded "enable and wait" path.

See [Architecture](docs/ARCHITECTURE.md) for the proposed contracts and
[SDF migration](docs/SDF_MIGRATION.md) for the prototype extraction plan.

## Scope

The initial release targets fixed-field extraction for practical runtime proxy
generation. Automatic differentiation, training-time regularizers,
tetrahedral output, adaptive grids, renderer materials, UV transfer, and GI
policy are not part of the first release.

FlexiCubes improves the mesh extracted from a field; it does not make
mesh-to-field conversion cheap by itself. The SDF builder and its bounded GPU
acceleration are therefore first-class parts of this project.

## Development and releases

This repository is the canonical source. Applications should consume immutable
commits during early integration and exact `0.x` package versions after the
first prerelease. Application repositories should keep only their adapter,
scheduling policy, cache ownership, and product-specific fallback logic.

The project will publish a package only after the CPU reference, WebGPU
implementation, attribution audit, and browser correctness suite agree on
golden fields and meshes. Until then, examples in documents are design
contracts rather than released API.

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
