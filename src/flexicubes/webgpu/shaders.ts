/*
 * FlexiCubes topology and dual-vertex stages are adapted from:
 * https://github.com/nv-tlabs/FlexiCubes
 *
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * Modified for flexicubes-webgpu as fixed-field WebGPU compute stages with
 * structured-grid indexing, bounded output, and explicit ownership.
 */

const FLEXICUBES_PARAMS_WGSL = /* wgsl */ `
struct Params {
  workStart: u32,
  workBatchCount: u32,
  totalCells: u32,
  maxQuads: u32,
  cellCountX: u32,
  cellCountY: u32,
  cellCountZ: u32,
  totalEdges: u32,
  sampleCountX: u32,
  sampleCountY: u32,
  sampleCountZ: u32,
  _padding0: u32,
  surface: vec4<f32>,
  sampleOrigin: vec4<f32>,
  sampleSpacing: vec4<f32>,
}

const CUBE_CORNERS = array<vec3<u32>, 8>(
  vec3<u32>(0u, 0u, 0u),
  vec3<u32>(1u, 0u, 0u),
  vec3<u32>(0u, 1u, 0u),
  vec3<u32>(1u, 1u, 0u),
  vec3<u32>(0u, 0u, 1u),
  vec3<u32>(1u, 0u, 1u),
  vec3<u32>(0u, 1u, 1u),
  vec3<u32>(1u, 1u, 1u)
);

fn cellCoordinates(index: u32) -> vec3<u32> {
  let x = index % params.cellCountX;
  let y = (index / params.cellCountX) % params.cellCountY;
  let z = index / (params.cellCountX * params.cellCountY);
  return vec3<u32>(x, y, z);
}

fn cellIndexAt(coordinates: vec3<u32>) -> u32 {
  return coordinates.x + params.cellCountX * (
    coordinates.y + params.cellCountY * coordinates.z
  );
}

fn sampleIndexAt(coordinates: vec3<u32>) -> u32 {
  return coordinates.x + params.sampleCountX * (
    coordinates.y + params.sampleCountY * coordinates.z
  );
}
`;

export const CLASSIFY_FLEXICUBES_CELLS_WGSL = /* wgsl */ `
${FLEXICUBES_PARAMS_WGSL}

@group(0) @binding(0) var<storage, read> fieldValues: array<f32>;
@group(0) @binding(1) var<storage, read_write> rawCases: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.workBatchCount) {
    return;
  }
  let cellIndex = params.workStart + globalId.x;
  if (cellIndex >= params.totalCells) {
    return;
  }

  let cell = cellCoordinates(cellIndex);
  var caseId = 0u;
  for (var corner = 0u; corner < 8u; corner = corner + 1u) {
    let value = fieldValues[sampleIndexAt(cell + CUBE_CORNERS[corner])];
    if (value < params.surface.x) {
      caseId = caseId | (1u << corner);
    }
  }
  rawCases[cellIndex] = caseId;
}
`;

export const RESOLVE_FLEXICUBES_CASES_WGSL = /* wgsl */ `
${FLEXICUBES_PARAMS_WGSL}

@group(0) @binding(0) var<storage, read> rawCases: array<u32>;
@group(0) @binding(1) var<storage, read> ambiguityChecks: array<i32>;
@group(0) @binding(2) var<storage, read> dualVertexCounts: array<u32>;
@group(0) @binding(3) var<storage, read_write> resolvedCases: array<u32>;
@group(0) @binding(4) var<storage, read_write> cellVertexOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.workBatchCount) {
    return;
  }
  let cellIndex = params.workStart + globalId.x;
  if (cellIndex >= params.totalCells) {
    return;
  }

  let rawCase = rawCases[cellIndex];
  let checkOffset = rawCase * 5u;
  var resolvedCase = rawCase;
  if (ambiguityChecks[checkOffset] == 1) {
    let cell = vec3<i32>(cellCoordinates(cellIndex));
    let adjacent = cell + vec3<i32>(
      ambiguityChecks[checkOffset + 1u],
      ambiguityChecks[checkOffset + 2u],
      ambiguityChecks[checkOffset + 3u]
    );
    let dimensions = vec3<i32>(
      i32(params.cellCountX),
      i32(params.cellCountY),
      i32(params.cellCountZ)
    );
    if (all(adjacent >= vec3<i32>(0)) && all(adjacent < dimensions)) {
      let adjacentCase = rawCases[cellIndexAt(vec3<u32>(adjacent))];
      if (ambiguityChecks[adjacentCase * 5u] == 1) {
        resolvedCase = u32(ambiguityChecks[checkOffset + 4u]);
      }
    }
  }

  let dualVertexCount = dualVertexCounts[resolvedCase];
  resolvedCases[cellIndex] = resolvedCase;
  cellVertexOffsets[cellIndex] = atomicAdd(&counters[0], dualVertexCount);
  if (dualVertexCount > 0u) {
    atomicAdd(&counters[1], 1u);
  }
}
`;

export const BUILD_FLEXICUBES_DUAL_VERTICES_WGSL = /* wgsl */ `
${FLEXICUBES_PARAMS_WGSL}

const CUBE_EDGES = array<vec2<u32>, 12>(
  vec2<u32>(0u, 1u),
  vec2<u32>(1u, 5u),
  vec2<u32>(4u, 5u),
  vec2<u32>(0u, 4u),
  vec2<u32>(2u, 3u),
  vec2<u32>(3u, 7u),
  vec2<u32>(6u, 7u),
  vec2<u32>(2u, 6u),
  vec2<u32>(2u, 0u),
  vec2<u32>(3u, 1u),
  vec2<u32>(7u, 5u),
  vec2<u32>(6u, 4u)
);

@group(0) @binding(0) var<storage, read> fieldValues: array<f32>;
@group(0) @binding(1) var<storage, read> resolvedCases: array<u32>;
@group(0) @binding(2) var<storage, read> cellVertexOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> dmcTable: array<i32>;
@group(0) @binding(4) var<storage, read> dualVertexCounts: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputPositions: array<f32>;
@group(0) @binding(6) var<storage, read_write> sourceCells: array<u32>;
@group(0) @binding(7) var<storage, read> params: Params;

fn cornerPosition(cell: vec3<u32>, corner: u32) -> vec3<f32> {
  return params.sampleOrigin.xyz + vec3<f32>(
    cell + CUBE_CORNERS[corner]
  ) * params.sampleSpacing.xyz;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.workBatchCount) {
    return;
  }
  let cellIndex = params.workStart + globalId.x;
  if (cellIndex >= params.totalCells) {
    return;
  }

  let caseId = resolvedCases[cellIndex];
  let dualVertexCount = dualVertexCounts[caseId];
  if (dualVertexCount == 0u) {
    return;
  }
  let cell = cellCoordinates(cellIndex);
  let caseOffset = caseId * 28u;
  let outputOffset = cellVertexOffsets[cellIndex];

  for (
    var dualVertex = 0u;
    dualVertex < dualVertexCount;
    dualVertex = dualVertex + 1u
  ) {
    var positionSum = vec3<f32>(0.0);
    var crossingCount = 0u;
    for (var slot = 0u; slot < 7u; slot = slot + 1u) {
      let localEdge = dmcTable[caseOffset + dualVertex * 7u + slot];
      if (localEdge < 0) {
        continue;
      }
      let edge = CUBE_EDGES[u32(localEdge)];
      let startSample = sampleIndexAt(cell + CUBE_CORNERS[edge.x]);
      let endSample = sampleIndexAt(cell + CUBE_CORNERS[edge.y]);
      let startValue = fieldValues[startSample];
      let endValue = fieldValues[endSample];
      let t = (params.surface.x - startValue) / (endValue - startValue);
      let start = cornerPosition(cell, edge.x);
      let end = cornerPosition(cell, edge.y);
      positionSum = positionSum + mix(start, end, t);
      crossingCount = crossingCount + 1u;
    }

    let vertexIndex = outputOffset + dualVertex;
    let position = positionSum / f32(crossingCount);
    let positionOffset = vertexIndex * 3u;
    outputPositions[positionOffset] = position.x;
    outputPositions[positionOffset + 1u] = position.y;
    outputPositions[positionOffset + 2u] = position.z;
    sourceCells[vertexIndex] = cellIndex;
  }
}
`;

export const TRIANGULATE_FLEXICUBES_WGSL = /* wgsl */ `
${FLEXICUBES_PARAMS_WGSL}

@group(0) @binding(0) var<storage, read> fieldValues: array<f32>;
@group(0) @binding(1) var<storage, read> resolvedCases: array<u32>;
@group(0) @binding(2) var<storage, read> cellVertexOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> dmcTable: array<i32>;
@group(0) @binding(4) var<storage, read_write> outputIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> params: Params;

fn dualGroupForEdge(caseId: u32, localEdge: u32) -> u32 {
  let caseOffset = caseId * 28u;
  for (var group = 0u; group < 4u; group = group + 1u) {
    for (var slot = 0u; slot < 7u; slot = slot + 1u) {
      if (dmcTable[caseOffset + group * 7u + slot] == i32(localEdge)) {
        return group;
      }
    }
  }
  return 0xffffffffu;
}

fn vertexForCellEdge(cell: vec3<u32>, localEdge: u32) -> u32 {
  let index = cellIndexAt(cell);
  let group = dualGroupForEdge(resolvedCases[index], localEdge);
  return cellVertexOffsets[index] + group;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.workBatchCount) {
    return;
  }
  let globalEdgeIndex = params.workStart + globalId.x;
  if (globalEdgeIndex >= params.totalEdges) {
    return;
  }

  let xEdgeCount = params.cellCountX * params.sampleCountY * params.sampleCountZ;
  let yEdgeCount = params.cellCountY * params.sampleCountX * params.sampleCountZ;
  var axis = 0u;
  var start = vec3<u32>(0u);
  var end = vec3<u32>(0u);
  var cells: array<vec3<u32>, 4>;
  var localEdges: array<u32, 4>;
  var interior = false;

  if (globalEdgeIndex < xEdgeCount) {
    let localIndex = globalEdgeIndex;
    let x = localIndex % params.cellCountX;
    let remainder = localIndex / params.cellCountX;
    let y = remainder % params.sampleCountY;
    let z = remainder / params.sampleCountY;
    axis = 0u;
    start = vec3<u32>(x, y, z);
    end = vec3<u32>(x + 1u, y, z);
    interior = y > 0u && y < params.cellCountY && z > 0u && z < params.cellCountZ;
    if (interior) {
      cells[0] = vec3<u32>(x, y - 1u, z - 1u);
      cells[1] = vec3<u32>(x, y, z - 1u);
      cells[2] = vec3<u32>(x, y, z);
      cells[3] = vec3<u32>(x, y - 1u, z);
      localEdges = array<u32, 4>(6u, 2u, 0u, 4u);
    }
  } else if (globalEdgeIndex < xEdgeCount + yEdgeCount) {
    let localIndex = globalEdgeIndex - xEdgeCount;
    let y = localIndex % params.cellCountY;
    let remainder = localIndex / params.cellCountY;
    let x = remainder % params.sampleCountX;
    let z = remainder / params.sampleCountX;
    axis = 1u;
    start = vec3<u32>(x, y, z);
    end = vec3<u32>(x, y + 1u, z);
    interior = x > 0u && x < params.cellCountX && z > 0u && z < params.cellCountZ;
    if (interior) {
      cells[0] = vec3<u32>(x - 1u, y, z - 1u);
      cells[1] = vec3<u32>(x, y, z - 1u);
      cells[2] = vec3<u32>(x, y, z);
      cells[3] = vec3<u32>(x - 1u, y, z);
      localEdges = array<u32, 4>(10u, 11u, 8u, 9u);
    }
  } else {
    let localIndex = globalEdgeIndex - xEdgeCount - yEdgeCount;
    let z = localIndex % params.cellCountZ;
    let remainder = localIndex / params.cellCountZ;
    let x = remainder % params.sampleCountX;
    let y = remainder / params.sampleCountX;
    axis = 2u;
    start = vec3<u32>(x, y, z);
    end = vec3<u32>(x, y, z + 1u);
    interior = x > 0u && x < params.cellCountX && y > 0u && y < params.cellCountY;
    if (interior) {
      cells[0] = vec3<u32>(x - 1u, y - 1u, z);
      cells[1] = vec3<u32>(x, y - 1u, z);
      cells[2] = vec3<u32>(x, y, z);
      cells[3] = vec3<u32>(x - 1u, y, z);
      localEdges = array<u32, 4>(5u, 7u, 3u, 1u);
    }
  }

  let startValue = fieldValues[sampleIndexAt(start)];
  let endValue = fieldValues[sampleIndexAt(end)];
  let startInside = startValue < params.surface.x;
  let endInside = endValue < params.surface.x;
  if (startInside == endInside) {
    return;
  }
  if (!interior) {
    atomicAdd(&counters[2], 1u);
    return;
  }

  var quad = array<u32, 4>(
    vertexForCellEdge(cells[0], localEdges[0]),
    vertexForCellEdge(cells[1], localEdges[1]),
    vertexForCellEdge(cells[2], localEdges[2]),
    vertexForCellEdge(cells[3], localEdges[3])
  );
  let baseNormalIsPositiveAxis = axis != 1u;
  let positiveFieldIsPositiveAxis = startInside;
  if (baseNormalIsPositiveAxis != positiveFieldIsPositiveAxis) {
    quad = array<u32, 4>(quad[0], quad[3], quad[2], quad[1]);
  }

  let quadIndex = atomicAdd(&counters[0], 1u);
  if (quadIndex >= params.maxQuads) {
    atomicStore(&counters[1], 1u);
    return;
  }
  let outputOffset = quadIndex * 6u;
  outputIndices[outputOffset] = quad[0];
  outputIndices[outputOffset + 1u] = quad[1];
  outputIndices[outputOffset + 2u] = quad[3];
  outputIndices[outputOffset + 3u] = quad[3];
  outputIndices[outputOffset + 4u] = quad[1];
  outputIndices[outputOffset + 5u] = quad[2];
}
`;
