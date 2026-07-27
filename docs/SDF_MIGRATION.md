# SDF prototype migration

## Outcome

The reusable algorithms were reimplemented here without copying the Vutify
application implementation wholesale.

The Vutify prototype proved that WebGPU compute, dense binning, sparse brick
storage, and Babylon integration can work. It also contains assumptions that
should not become public contracts:

- Babylon.js compute and storage-buffer ownership
- Vutify logging, VFX resource, and lifecycle types
- synchronous JavaScript triangle merging and bin construction
- a nearest-triangle-normal sign heuristic
- cell-centered `gridSize^3` samples
- mandatory GPU-to-CPU readback after SDF construction

FlexiCubes does not remove those bottlenecks. The migration is worthwhile only
if the public pipeline fixes them.

## Prototype inventory

The first extraction should treat the prototype as a set of algorithmic
references:

| Prototype component | Reusable idea | Migration decision |
|---|---|---|
| Dense CPU SDF | bounds, domain fitting, memory/work estimates, distance reference | Port pure math with tests; separate logging and change the field to vertex samples |
| Sparse planner | sparse plan, packed triangle buffers, brick headers, size estimates | Port typed planning helpers; replace nested-array bin building for production |
| Dense GPU SDF | binned distance WGSL and dispatch layout | Adapt WGSL after lattice/sign corrections; replace renderer resource ownership |
| Sparse GPU SDF | sparse brick and fallback-field compute | Adapt after the dense reference is correct; preserve a conservative brick halo |

Any source actually adapted from Vutify should state its origin and
modification. Any source adapted from NVIDIA's implementation must additionally
retain NVIDIA's applicable Apache-2.0 notice.

## Migration phases

### Phase 1: contracts and CPU oracle — complete

Deliver:

- renderer-neutral mesh, domain, scalar-field, and result contracts
- vertex-sampled dense grid construction
- analytic CPU distance and explicit `parity`, `parity-shell-union`, and
  `shell` sign references
- fixtures for cube, sphere, plane, thin wall, non-convex closed mesh, and
  reversed winding

Accept when:

- field indexing and domain endpoints are proven by tests
- sign policy is explicit in every result
- no Vutify or Babylon imports exist

### Phase 2: bounded dense WebGPU SDF — complete

Deliver:

- raw WebGPU buffer ownership
- GPU triangle preprocessing and bin construction
- tiled distance and sign dispatches
- progress, cancellation, device-limit checks, and memory estimates

Accept when:

- the UI thread has no triangle-count-proportional long task
- CPU and GPU fixtures agree within tolerance
- a one-million-triangle input returns control immediately and either completes
  inside its budget or refuses before allocation

The existing Vutify distance WGSL is useful here. Its CPU bin builder and sign
line are not.

### Phase 3: fixed-field FlexiCubes CPU reference — complete

Deliver:

- Apache-attributed lookup tables
- regular-grid topology classification
- dual-vertex computation and triangular output
- provenance sufficient for attribute transfer
- golden comparisons with the official PyTorch implementation

Accept when:

- analytic fields match expected orientation and topology
- ambiguous cases match the reference
- fixed-field limitations are documented without claiming training-time quality

### Phase 4: composed GPU extraction — complete

Deliver:

- GPU surface-cell classification
- bounded atomic output allocation
- GPU dual vertices and triangulation
- SDF-to-extractor handoff without CPU field readback
- bounded output triangle count

Accept when:

- the composed path is measurably faster than downloading the field
- cancellation frees all owned buffers
- output matches the CPU oracle within tolerance

### Phase 5: sparse fields and production hardening — pending

Deliver:

- sparse active-brick construction with conservative halo
- extraction across brick boundaries
- browser/device benchmark matrix
- device loss, memory pressure, and unsupported-WebGPU behavior

Accept when:

- dense and sparse output agree at brick boundaries
- peak memory and long-task budgets are reported
- no job can block application rendering while it builds

## Vutify integration

During early development:

1. Keep this public repository as the canonical implementation.
2. Develop it as a sibling checkout of Vutify.
3. Test Vutify against an immutable Git commit or a locally packed artifact.
4. Use an uncommitted local link only for rapid iteration.
5. Keep the Babylon adapter, GI scheduling, cache keys, materials, and fallback
   policy in Vutify.

After the first prerelease:

1. Publish an exact `0.x` package version.
2. Add it to Vutify's approved dependency registry and package manifest in a
   dedicated integration change.
3. Pin the exact version; do not consume a moving branch.
4. Upgrade with an explicit changelog and Vutify proxy regression fixtures.

A Git submodule is not the default. It preserves separate history but adds
clone, CI, and pointer-update failure modes. A subtree is worse because it
creates two apparent sources of truth. Immutable package versions or Git SHAs
keep the public repository canonical with less Vutify-specific machinery.

## Vutify fallback policy

The library builds proxies; Vutify decides when they are needed.

- Simple primitives, walls, and already-low-poly meshes use direct geometry.
- Complex static meshes request a background SDF/FlexiCubes proxy.
- Animated/skinned meshes keep the last valid proxy or a conservative fallback
  until an explicitly budgeted refresh finishes.
- If proxy generation refuses or times out, GI continues with direct lighting
  and the best available conservative representation. Enabling GI must never
  block the scene.

This makes FlexiCubes useful for the complex-mesh problem without forcing every
object through it.
