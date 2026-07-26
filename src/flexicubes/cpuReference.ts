/*
 * FlexiCubes algorithm structure in this file is adapted from:
 * https://github.com/nv-tlabs/FlexiCubes
 *
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * Modified for flexicubes-webgpu:
 * - rewritten from PyTorch tensor operations to renderer-neutral TypeScript
 * - fixed-field triangular extraction only
 * - structured x-fastest lattice input
 * - explicit output bounds, provenance, cancellation, and progress contracts
 * - geometric quad ordering independent of tensor sort implementation
 */

import type {
  DenseScalarField3D,
  FlexiCubesExtractOptions,
  FlexiCubesExtractionResult,
  FlexiCubesPhase,
  Vec3,
} from "../core/types.js";
import {
  FLEXICUBES_AMBIGUITY_CHECKS,
  FLEXICUBES_DMC_TABLE,
  FLEXICUBES_DUAL_VERTEX_COUNTS,
  FLEXICUBES_EDGES_PER_DUAL_VERTEX,
  FLEXICUBES_MAX_DUAL_VERTICES,
} from "./tables.js";

const CUBE_CORNERS: readonly Vec3[] = [
  [0, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [1, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [1, 1, 1],
];

const CUBE_EDGES: readonly (readonly [start: number, end: number])[] = [
  [0, 1],
  [1, 5],
  [4, 5],
  [0, 4],
  [2, 3],
  [3, 7],
  [6, 7],
  [2, 6],
  [2, 0],
  [3, 1],
  [7, 5],
  [6, 4],
];

interface SurfaceCell {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly sourceCellIndex: number;
  readonly caseId: number;
  readonly sampleIndices: Uint32Array;
  readonly edgeIds: Int32Array;
  readonly dualVertexByEdge: Int32Array;
}

interface EdgeOccurrence {
  readonly surfaceCellIndex: number;
  readonly localEdgeIndex: number;
}

interface SurfaceEdge {
  readonly id: number;
  readonly startSample: number;
  readonly endSample: number;
  readonly occurrences: EdgeOccurrence[];
  crossing: Vec3 | undefined;
}

export class FlexiCubesAbortError extends Error {
  public override readonly name = "AbortError";

  public constructor() {
    super("FlexiCubes extraction was aborted");
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw new FlexiCubesAbortError();
  }
}

function reportProgress(
  options: FlexiCubesExtractOptions,
  phase: FlexiCubesPhase,
  completed: number,
  total: number,
): void {
  options.onProgress?.({
    phase,
    completed,
    total,
    fraction: total === 0 ? 1 : completed / total,
  });
}

function validateField(field: DenseScalarField3D): void {
  if (field.signConvention !== "negative-inside") {
    throw new RangeError(
      "FlexiCubes requires a negative-inside scalar field",
    );
  }
  for (let axis = 0; axis < 3; axis++) {
    if (field.sampleCounts[axis] !== field.cellCounts[axis]! + 1) {
      throw new RangeError(
        `sampleCounts[${axis}] must equal cellCounts[${axis}] + 1`,
      );
    }
    if (!(field.sampleSpacing[axis]! > 0)) {
      throw new RangeError(`sampleSpacing[${axis}] must be positive`);
    }
  }
  if (field.values.length !== field.sampleCount) {
    throw new RangeError("field values do not match sampleCount");
  }
  const expectedCount = (
    field.sampleCounts[0]
    * field.sampleCounts[1]
    * field.sampleCounts[2]
  );
  if (field.sampleCount !== expectedCount) {
    throw new RangeError("sampleCount does not match sampleCounts");
  }
}

function cellIndex(
  field: DenseScalarField3D,
  x: number,
  y: number,
  z: number,
): number {
  return (
    x
    + field.cellCounts[0] * (y + field.cellCounts[1] * z)
  );
}

function sampleIndex(
  field: DenseScalarField3D,
  x: number,
  y: number,
  z: number,
): number {
  return (
    x
    + field.sampleCounts[0] * (y + field.sampleCounts[1] * z)
  );
}

function cellSampleIndices(
  field: DenseScalarField3D,
  x: number,
  y: number,
  z: number,
): Uint32Array {
  const indices = new Uint32Array(8);
  for (let corner = 0; corner < CUBE_CORNERS.length; corner++) {
    const offset = CUBE_CORNERS[corner]!;
    indices[corner] = sampleIndex(
      field,
      x + offset[0],
      y + offset[1],
      z + offset[2],
    );
  }
  return indices;
}

function classifyCell(
  field: DenseScalarField3D,
  indices: Uint32Array,
  isoValue: number,
): number {
  let caseId = 0;
  for (let corner = 0; corner < 8; corner++) {
    if (field.values[indices[corner]!]! < isoValue) {
      caseId |= 1 << corner;
    }
  }
  return caseId;
}

function ambiguityEntry(caseId: number, component: number): number {
  return FLEXICUBES_AMBIGUITY_CHECKS[caseId * 5 + component]!;
}

function resolveAmbiguousCase(
  field: DenseScalarField3D,
  rawCases: Uint8Array,
  x: number,
  y: number,
  z: number,
  rawCase: number,
): number {
  if (ambiguityEntry(rawCase, 0) !== 1) {
    return rawCase;
  }

  const adjacentX = x + ambiguityEntry(rawCase, 1);
  const adjacentY = y + ambiguityEntry(rawCase, 2);
  const adjacentZ = z + ambiguityEntry(rawCase, 3);
  if (
    adjacentX < 0
    || adjacentY < 0
    || adjacentZ < 0
    || adjacentX >= field.cellCounts[0]
    || adjacentY >= field.cellCounts[1]
    || adjacentZ >= field.cellCounts[2]
  ) {
    return rawCase;
  }

  const adjacentCase = rawCases[
    cellIndex(field, adjacentX, adjacentY, adjacentZ)
  ]!;
  return ambiguityEntry(adjacentCase, 0) === 1
    ? ambiguityEntry(rawCase, 4)
    : rawCase;
}

function identifySurfaceCells(
  field: DenseScalarField3D,
  isoValue: number,
  options: FlexiCubesExtractOptions,
): SurfaceCell[] {
  const totalCells = (
    field.cellCounts[0] * field.cellCounts[1] * field.cellCounts[2]
  );
  if (totalCells > 0xffff_ffff) {
    throw new RangeError("cell count exceeds Uint32 provenance indexing");
  }
  const rawCases = new Uint8Array(totalCells);
  let completed = 0;
  reportProgress(options, "classify-cells", completed, totalCells);
  for (let z = 0; z < field.cellCounts[2]; z++) {
    for (let y = 0; y < field.cellCounts[1]; y++) {
      for (let x = 0; x < field.cellCounts[0]; x++) {
        if ((completed & 4095) === 0) {
          throwIfAborted(options.signal);
        }
        const indices = cellSampleIndices(field, x, y, z);
        rawCases[cellIndex(field, x, y, z)] = classifyCell(
          field,
          indices,
          isoValue,
        );
        completed++;
      }
    }
    reportProgress(options, "classify-cells", completed, totalCells);
  }

  const surfaceCells: SurfaceCell[] = [];
  for (let x = 0; x < field.cellCounts[0]; x++) {
    for (let y = 0; y < field.cellCounts[1]; y++) {
      for (let z = 0; z < field.cellCounts[2]; z++) {
        const sourceCellIndex = cellIndex(field, x, y, z);
        const rawCase = rawCases[sourceCellIndex]!;
        if (rawCase === 0 || rawCase === 255) {
          continue;
        }
        const edgeIds = new Int32Array(12);
        const dualVertexByEdge = new Int32Array(12);
        edgeIds.fill(-1);
        dualVertexByEdge.fill(-1);
        surfaceCells.push({
          x,
          y,
          z,
          sourceCellIndex,
          caseId: resolveAmbiguousCase(
            field,
            rawCases,
            x,
            y,
            z,
            rawCase,
          ),
          sampleIndices: cellSampleIndices(field, x, y, z),
          edgeIds,
          dualVertexByEdge,
        });
      }
    }
  }
  return surfaceCells;
}

function edgeKey(start: number, end: number): string {
  return start < end ? `${start}:${end}` : `${end}:${start}`;
}

function samplePosition(
  field: DenseScalarField3D,
  index: number,
): Vec3 {
  const x = index % field.sampleCounts[0];
  const y = Math.floor(index / field.sampleCounts[0])
    % field.sampleCounts[1];
  const z = Math.floor(
    index / (field.sampleCounts[0] * field.sampleCounts[1]),
  );
  return [
    field.sampleOrigin[0] + x * field.sampleSpacing[0],
    field.sampleOrigin[1] + y * field.sampleSpacing[1],
    field.sampleOrigin[2] + z * field.sampleSpacing[2],
  ];
}

function interpolateCrossing(
  field: DenseScalarField3D,
  edge: SurfaceEdge,
  isoValue: number,
): Vec3 {
  const startValue = field.values[edge.startSample]!;
  const endValue = field.values[edge.endSample]!;
  const denominator = endValue - startValue;
  if (denominator === 0) {
    throw new Error("surface edge has equal endpoint values");
  }
  const t = (isoValue - startValue) / denominator;
  const start = samplePosition(field, edge.startSample);
  const end = samplePosition(field, edge.endSample);
  return [
    start[0] + t * (end[0] - start[0]),
    start[1] + t * (end[1] - start[1]),
    start[2] + t * (end[2] - start[2]),
  ];
}

function identifySurfaceEdges(
  field: DenseScalarField3D,
  surfaceCells: readonly SurfaceCell[],
  isoValue: number,
  options: FlexiCubesExtractOptions,
): SurfaceEdge[] {
  const edgeByKey = new Map<string, SurfaceEdge>();
  const edges: SurfaceEdge[] = [];
  reportProgress(
    options,
    "build-surface-edges",
    0,
    surfaceCells.length,
  );

  for (
    let surfaceCellIndex = 0;
    surfaceCellIndex < surfaceCells.length;
    surfaceCellIndex++
  ) {
    throwIfAborted(options.signal);
    const cell = surfaceCells[surfaceCellIndex]!;
    for (let localEdgeIndex = 0; localEdgeIndex < CUBE_EDGES.length; localEdgeIndex++) {
      const localEdge = CUBE_EDGES[localEdgeIndex]!;
      const startSample = cell.sampleIndices[localEdge[0]]!;
      const endSample = cell.sampleIndices[localEdge[1]]!;
      const startInside = field.values[startSample]! < isoValue;
      const endInside = field.values[endSample]! < isoValue;
      if (startInside === endInside) {
        continue;
      }

      const key = edgeKey(startSample, endSample);
      let edge = edgeByKey.get(key);
      if (edge === undefined) {
        edge = {
          id: edges.length,
          startSample,
          endSample,
          occurrences: [],
          crossing: undefined,
        };
        edgeByKey.set(key, edge);
        edges.push(edge);
      }
      edge.occurrences.push({ surfaceCellIndex, localEdgeIndex });
      cell.edgeIds[localEdgeIndex] = edge.id;
    }
    reportProgress(
      options,
      "build-surface-edges",
      surfaceCellIndex + 1,
      surfaceCells.length,
    );
  }

  for (const edge of edges) {
    edge.crossing = interpolateCrossing(field, edge, isoValue);
  }
  return edges;
}

function dmcEdge(
  caseId: number,
  dualVertex: number,
  slot: number,
): number {
  const caseStride = (
    FLEXICUBES_MAX_DUAL_VERTICES
    * FLEXICUBES_EDGES_PER_DUAL_VERTEX
  );
  return FLEXICUBES_DMC_TABLE[
    caseId * caseStride
    + dualVertex * FLEXICUBES_EDGES_PER_DUAL_VERTEX
    + slot
  ]!;
}

function requireCrossing(edge: SurfaceEdge): Vec3 {
  if (edge.crossing === undefined) {
    throw new Error("surface edge crossing was not initialized");
  }
  return edge.crossing;
}

function buildDualVertices(
  surfaceCells: readonly SurfaceCell[],
  edges: readonly SurfaceEdge[],
  options: FlexiCubesExtractOptions,
): { positions: number[]; sourceCells: number[] } {
  const positions: number[] = [];
  const sourceCells: number[] = [];
  reportProgress(
    options,
    "build-dual-vertices",
    0,
    surfaceCells.length,
  );

  for (let cellIndex = 0; cellIndex < surfaceCells.length; cellIndex++) {
    throwIfAborted(options.signal);
    const cell = surfaceCells[cellIndex]!;
    const dualVertexCount = FLEXICUBES_DUAL_VERTEX_COUNTS[cell.caseId]!;
    for (
      let localDualVertex = 0;
      localDualVertex < dualVertexCount;
      localDualVertex++
    ) {
      let sumX = 0;
      let sumY = 0;
      let sumZ = 0;
      let crossingCount = 0;
      const vertexIndex = positions.length / 3;
      for (
        let slot = 0;
        slot < FLEXICUBES_EDGES_PER_DUAL_VERTEX;
        slot++
      ) {
        const localEdgeIndex = dmcEdge(
          cell.caseId,
          localDualVertex,
          slot,
        );
        if (localEdgeIndex < 0) {
          continue;
        }
        const edgeId = cell.edgeIds[localEdgeIndex]!;
        if (edgeId < 0) {
          throw new Error(
            `FlexiCubes case ${cell.caseId} references a non-crossing edge`,
          );
        }
        const crossing = requireCrossing(edges[edgeId]!);
        sumX += crossing[0];
        sumY += crossing[1];
        sumZ += crossing[2];
        crossingCount++;
        cell.dualVertexByEdge[localEdgeIndex] = vertexIndex;
      }
      if (crossingCount === 0) {
        throw new Error(
          `FlexiCubes case ${cell.caseId} emitted an empty dual vertex`,
        );
      }
      positions.push(
        sumX / crossingCount,
        sumY / crossingCount,
        sumZ / crossingCount,
      );
      sourceCells.push(cell.sourceCellIndex);
    }
    reportProgress(
      options,
      "build-dual-vertices",
      cellIndex + 1,
      surfaceCells.length,
    );
  }
  return { positions, sourceCells };
}

function vertexPosition(positions: readonly number[], index: number): Vec3 {
  const offset = index * 3;
  return [
    positions[offset]!,
    positions[offset + 1]!,
    positions[offset + 2]!,
  ];
}

function dot(left: Vec3, right: Vec3): number {
  return (
    left[0] * right[0]
    + left[1] * right[1]
    + left[2] * right[2]
  );
}

function subtract(left: Vec3, right: Vec3): Vec3 {
  return [
    left[0] - right[0],
    left[1] - right[1],
    left[2] - right[2],
  ];
}

function cross(left: Vec3, right: Vec3): Vec3 {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function orderQuadAroundEdge(
  field: DenseScalarField3D,
  edge: SurfaceEdge,
  quad: readonly number[],
  positions: readonly number[],
  isoValue: number,
): number[] {
  const edgeStart = samplePosition(field, edge.startSample);
  const edgeEnd = samplePosition(field, edge.endSample);
  const edgeDirection = subtract(edgeEnd, edgeStart);
  const midpoint: Vec3 = [
    (edgeStart[0] + edgeEnd[0]) * 0.5,
    (edgeStart[1] + edgeEnd[1]) * 0.5,
    (edgeStart[2] + edgeEnd[2]) * 0.5,
  ];
  const axis = Math.abs(edgeDirection[0]) > 0
    ? 0
    : Math.abs(edgeDirection[1]) > 0
      ? 1
      : 2;
  const ordered = [...quad].sort((leftIndex, rightIndex) => {
    const left = subtract(vertexPosition(positions, leftIndex), midpoint);
    const right = subtract(vertexPosition(positions, rightIndex), midpoint);
    const leftAngle = axis === 0
      ? Math.atan2(left[2], left[1])
      : axis === 1
        ? Math.atan2(left[2], left[0])
        : Math.atan2(left[1], left[0]);
    const rightAngle = axis === 0
      ? Math.atan2(right[2], right[1])
      : axis === 1
        ? Math.atan2(right[2], right[0])
        : Math.atan2(right[1], right[0]);
    return leftAngle - rightAngle;
  });

  const a = vertexPosition(positions, ordered[0]!);
  const b = vertexPosition(positions, ordered[1]!);
  const d = vertexPosition(positions, ordered[3]!);
  const normal = cross(subtract(b, a), subtract(d, a));
  const positiveDirection = field.values[edge.startSample]! < isoValue
    ? edgeDirection
    : subtract(edgeStart, edgeEnd);
  if (dot(normal, positiveDirection) < 0) {
    return [ordered[0]!, ordered[3]!, ordered[2]!, ordered[1]!];
  }
  return ordered;
}

function fieldGradientAt(
  field: DenseScalarField3D,
  position: Vec3,
): Vec3 {
  const gridX = (
    position[0] - field.sampleOrigin[0]
  ) / field.sampleSpacing[0];
  const gridY = (
    position[1] - field.sampleOrigin[1]
  ) / field.sampleSpacing[1];
  const gridZ = (
    position[2] - field.sampleOrigin[2]
  ) / field.sampleSpacing[2];
  const x = Math.max(
    0,
    Math.min(field.cellCounts[0] - 1, Math.floor(gridX)),
  );
  const y = Math.max(
    0,
    Math.min(field.cellCounts[1] - 1, Math.floor(gridY)),
  );
  const z = Math.max(
    0,
    Math.min(field.cellCounts[2] - 1, Math.floor(gridZ)),
  );
  const tx = Math.max(0, Math.min(1, gridX - x));
  const ty = Math.max(0, Math.min(1, gridY - y));
  const tz = Math.max(0, Math.min(1, gridZ - z));
  const value = (dx: number, dy: number, dz: number): number => (
    field.values[sampleIndex(field, x + dx, y + dy, z + dz)]!
  );

  const gradientX = (
    (
      (value(1, 0, 0) - value(0, 0, 0)) * (1 - ty)
      + (value(1, 1, 0) - value(0, 1, 0)) * ty
    ) * (1 - tz)
    + (
      (value(1, 0, 1) - value(0, 0, 1)) * (1 - ty)
      + (value(1, 1, 1) - value(0, 1, 1)) * ty
    ) * tz
  ) / field.sampleSpacing[0];
  const gradientY = (
    (
      (value(0, 1, 0) - value(0, 0, 0)) * (1 - tx)
      + (value(1, 1, 0) - value(1, 0, 0)) * tx
    ) * (1 - tz)
    + (
      (value(0, 1, 1) - value(0, 0, 1)) * (1 - tx)
      + (value(1, 1, 1) - value(1, 0, 1)) * tx
    ) * tz
  ) / field.sampleSpacing[1];
  const gradientZ = (
    (
      (value(0, 0, 1) - value(0, 0, 0)) * (1 - tx)
      + (value(1, 0, 1) - value(1, 0, 0)) * tx
    ) * (1 - ty)
    + (
      (value(0, 1, 1) - value(0, 1, 0)) * (1 - tx)
      + (value(1, 1, 1) - value(1, 1, 0)) * tx
    ) * ty
  ) / field.sampleSpacing[2];
  return [gradientX, gradientY, gradientZ];
}

function orientTriangleToPositiveField(
  field: DenseScalarField3D,
  positions: readonly number[],
  aIndex: number,
  bIndex: number,
  cIndex: number,
): readonly [number, number, number] {
  const a = vertexPosition(positions, aIndex);
  const b = vertexPosition(positions, bIndex);
  const c = vertexPosition(positions, cIndex);
  const normal = cross(subtract(b, a), subtract(c, a));
  const center: Vec3 = [
    (a[0] + b[0] + c[0]) / 3,
    (a[1] + b[1] + c[1]) / 3,
    (a[2] + b[2] + c[2]) / 3,
  ];
  return dot(normal, fieldGradientAt(field, center)) < 0
    ? [aIndex, cIndex, bIndex]
    : [aIndex, bIndex, cIndex];
}

function triangulate(
  field: DenseScalarField3D,
  surfaceCells: readonly SurfaceCell[],
  edges: readonly SurfaceEdge[],
  positions: readonly number[],
  isoValue: number,
  maxOutputTriangles: number,
  options: FlexiCubesExtractOptions,
): {
  readonly indices: number[];
  readonly boundarySurfaceEdgeCount: number;
  readonly quadCount: number;
} {
  const indices: number[] = [];
  let boundarySurfaceEdgeCount = 0;
  let quadCount = 0;
  reportProgress(options, "triangulate", 0, edges.length);

  for (let edgeIndex = 0; edgeIndex < edges.length; edgeIndex++) {
    throwIfAborted(options.signal);
    const edge = edges[edgeIndex]!;
    if (edge.occurrences.length !== 4) {
      boundarySurfaceEdgeCount++;
      reportProgress(options, "triangulate", edgeIndex + 1, edges.length);
      continue;
    }

    const quad = edge.occurrences.map((occurrence) => {
      const vertexIndex = surfaceCells[
        occurrence.surfaceCellIndex
      ]!.dualVertexByEdge[occurrence.localEdgeIndex]!;
      if (vertexIndex < 0) {
        throw new Error("surface edge was not assigned a dual vertex");
      }
      return vertexIndex;
    });
    const ordered = orderQuadAroundEdge(
      field,
      edge,
      quad,
      positions,
      isoValue,
    );
    if (indices.length / 3 + 2 > maxOutputTriangles) {
      throw new RangeError(
        `FlexiCubes output exceeds ${maxOutputTriangles} triangles`,
      );
    }

    // Uniform gamma weights select the reference implementation's second
    // diagonal: [0, 1, 3] and [3, 1, 2].
    const firstTriangle = orientTriangleToPositiveField(
      field,
      positions,
      ordered[0]!,
      ordered[1]!,
      ordered[3]!,
    );
    const secondTriangle = orientTriangleToPositiveField(
      field,
      positions,
      ordered[3]!,
      ordered[1]!,
      ordered[2]!,
    );
    indices.push(...firstTriangle, ...secondTriangle);
    quadCount++;
    reportProgress(options, "triangulate", edgeIndex + 1, edges.length);
  }

  return { indices, boundarySurfaceEdgeCount, quadCount };
}

export function extractFlexiCubesCpuReference(
  field: DenseScalarField3D,
  options: FlexiCubesExtractOptions = {},
): FlexiCubesExtractionResult {
  validateField(field);
  throwIfAborted(options.signal);
  const isoValue = options.isoValue ?? 0;
  if (!Number.isFinite(isoValue)) {
    throw new RangeError("isoValue must be finite");
  }
  const maxOutputTriangles = options.maxOutputTriangles ?? 10_000_000;
  if (
    !Number.isSafeInteger(maxOutputTriangles)
    || maxOutputTriangles < 0
  ) {
    throw new RangeError(
      "maxOutputTriangles must be a non-negative safe integer",
    );
  }

  const startTime = performance.now();
  const surfaceCells = identifySurfaceCells(field, isoValue, options);
  if (surfaceCells.length === 0) {
    return {
      mesh: {
        positions: new Float32Array(),
        indices: new Uint32Array(),
        sourceCells: new Uint32Array(),
      },
      stats: {
        backend: "cpu-reference",
        surfaceCellCount: 0,
        surfaceEdgeCount: 0,
        boundarySurfaceEdgeCount: 0,
        dualVertexCount: 0,
        quadCount: 0,
        triangleCount: 0,
        elapsedMs: performance.now() - startTime,
      },
    };
  }

  const edges = identifySurfaceEdges(
    field,
    surfaceCells,
    isoValue,
    options,
  );
  const dualVertices = buildDualVertices(surfaceCells, edges, options);
  const faces = triangulate(
    field,
    surfaceCells,
    edges,
    dualVertices.positions,
    isoValue,
    maxOutputTriangles,
    options,
  );
  if (
    faces.boundarySurfaceEdgeCount > 0
    && options.allowBoundaryOpen !== true
  ) {
    throw new RangeError(
      `isosurface reaches the lattice boundary at `
      + `${faces.boundarySurfaceEdgeCount} surface edges`,
    );
  }

  return {
    mesh: {
      positions: new Float32Array(dualVertices.positions),
      indices: new Uint32Array(faces.indices),
      sourceCells: new Uint32Array(dualVertices.sourceCells),
    },
    stats: {
      backend: "cpu-reference",
      surfaceCellCount: surfaceCells.length,
      surfaceEdgeCount: edges.length,
      boundarySurfaceEdgeCount: faces.boundarySurfaceEdgeCount,
      dualVertexCount: dualVertices.positions.length / 3,
      quadCount: faces.quadCount,
      triangleCount: faces.indices.length / 3,
      elapsedMs: performance.now() - startTime,
    },
  };
}
