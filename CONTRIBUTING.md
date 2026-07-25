# Contributing

The project is being built in public from its first design boundary. Early
contributions should preserve that boundary rather than optimize for a single
engine or application.

## Before opening a change

- Discuss API or algorithm changes in an issue before implementing a large
  slice.
- Keep the core independent of Babylon.js, Three.js, Vutify, and framework
  lifecycle code.
- Do not add synchronous UI-thread work proportional to input triangle count or
  output grid volume.
- Add a deterministic CPU test before or alongside a WebGPU implementation.
- Include benchmark evidence for performance claims.
- Do not add a dependency without documenting why the browser and maintenance
  cost is justified.

## Source provenance

Contributions are provided under Apache-2.0.

When adapting source from NVIDIA's FlexiCubes reference implementation:

1. Confirm that the source is covered by its Apache-2.0 license.
2. Preserve the applicable NVIDIA copyright and license notice.
3. Add a prominent note that the file was modified.
4. Keep the repository-level `NOTICE` file with distributions.

Do not contribute code copied from a source with an incompatible or unclear
license.

## Correctness expectations

An extraction change should be tested against small analytic fields before
large visual examples:

- plane
- sphere
- axis-aligned box
- thin shell
- disconnected components
- empty and fully occupied fields
- ambiguous topology cases from the FlexiCubes reference tables

GPU results should be compared with the CPU reference using documented numeric
and topology tolerances. A visually plausible screenshot is supporting
evidence, not the correctness oracle.

## Commit scope

Prefer one independently reviewable boundary per commit: contracts, CPU
reference, a compute stage, tests, or integration. Renderer adapters belong in
optional adapters or application repositories, not in the core pipeline.
