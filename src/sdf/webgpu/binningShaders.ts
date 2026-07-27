/*
 * Adapted from Vutify's mesh-SDF prototype and rewritten for raw WebGPU.
 * The original renderer bindings and CPU bin construction are not included.
 */

const BINNING_COMMON_WGSL = /* wgsl */ `
struct CountParams {
  triangleStart: u32,
  triangleBatchCount: u32,
  triangleTotal: u32,
  distanceBinResolution: u32,
  parityBinResolution: u32,
  signMode: u32,
  _padding1: u32,
  _padding2: u32,
  domainMin: vec4<f32>,
  domainMax: vec4<f32>,
}

fn loadPosition(vertexIndex: u32) -> vec3<f32> {
  let offset = vertexIndex * 3u;
  return vec3<f32>(
    positions[offset],
    positions[offset + 1u],
    positions[offset + 2u]
  );
}

fn binCoordinate(
  value: f32,
  minimum: f32,
  maximum: f32,
  resolution: u32
) -> u32 {
  let normalized = clamp((value - minimum) / (maximum - minimum), 0.0, 0.99999994);
  return min(u32(floor(normalized * f32(resolution))), resolution - 1u);
}

fn overlapsDomain(triangleMin: vec3<f32>, triangleMax: vec3<f32>) -> bool {
  return !(
    triangleMax.x < params.domainMin.x ||
    triangleMax.y < params.domainMin.y ||
    triangleMax.z < params.domainMin.z ||
    triangleMin.x > params.domainMax.x ||
    triangleMin.y > params.domainMax.y ||
    triangleMin.z > params.domainMax.z
  );
}
`;

export const COUNT_TRIANGLE_BINS_WGSL = /* wgsl */ `
${BINNING_COMMON_WGSL}

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> distanceCounts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> parityCounts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> params: CountParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.triangleBatchCount) {
    return;
  }
  let triangleIndex = params.triangleStart + globalId.x;
  if (triangleIndex >= params.triangleTotal) {
    return;
  }

  let indexOffset = triangleIndex * 3u;
  let a = loadPosition(indices[indexOffset]);
  let b = loadPosition(indices[indexOffset + 1u]);
  let c = loadPosition(indices[indexOffset + 2u]);
  let triangleMin = min(a, min(b, c));
  let triangleMax = max(a, max(b, c));
  if (!overlapsDomain(triangleMin, triangleMax)) {
    return;
  }

  let distanceResolution = params.distanceBinResolution;
  let distanceMin = vec3<u32>(
    binCoordinate(triangleMin.x, params.domainMin.x, params.domainMax.x, distanceResolution),
    binCoordinate(triangleMin.y, params.domainMin.y, params.domainMax.y, distanceResolution),
    binCoordinate(triangleMin.z, params.domainMin.z, params.domainMax.z, distanceResolution)
  );
  let distanceMax = vec3<u32>(
    binCoordinate(triangleMax.x, params.domainMin.x, params.domainMax.x, distanceResolution),
    binCoordinate(triangleMax.y, params.domainMin.y, params.domainMax.y, distanceResolution),
    binCoordinate(triangleMax.z, params.domainMin.z, params.domainMax.z, distanceResolution)
  );
  for (var z = distanceMin.z; z <= distanceMax.z; z = z + 1u) {
    for (var y = distanceMin.y; y <= distanceMax.y; y = y + 1u) {
      for (var x = distanceMin.x; x <= distanceMax.x; x = x + 1u) {
        let binIndex = z * distanceResolution * distanceResolution + y * distanceResolution + x;
        atomicAdd(&distanceCounts[binIndex], 1u);
      }
    }
  }

  if (params.signMode != 1u && params.signMode != 3u) {
    return;
  }
  let parityResolution = params.parityBinResolution;
  let parityPlaneSize = parityResolution * parityResolution;
  let minX = binCoordinate(triangleMin.x, params.domainMin.x, params.domainMax.x, parityResolution);
  let minY = binCoordinate(triangleMin.y, params.domainMin.y, params.domainMax.y, parityResolution);
  let minZ = binCoordinate(triangleMin.z, params.domainMin.z, params.domainMax.z, parityResolution);
  let maxX = binCoordinate(triangleMax.x, params.domainMin.x, params.domainMax.x, parityResolution);
  let maxY = binCoordinate(triangleMax.y, params.domainMin.y, params.domainMax.y, parityResolution);
  let maxZ = binCoordinate(triangleMax.z, params.domainMin.z, params.domainMax.z, parityResolution);

  for (var z = minZ; z <= maxZ; z = z + 1u) {
    for (var y = minY; y <= maxY; y = y + 1u) {
      atomicAdd(&parityCounts[z * parityResolution + y], 1u);
    }
  }
  for (var z = minZ; z <= maxZ; z = z + 1u) {
    for (var x = minX; x <= maxX; x = x + 1u) {
      atomicAdd(&parityCounts[parityPlaneSize + z * parityResolution + x], 1u);
    }
  }
  for (var y = minY; y <= maxY; y = y + 1u) {
    for (var x = minX; x <= maxX; x = x + 1u) {
      atomicAdd(&parityCounts[2u * parityPlaneSize + y * parityResolution + x], 1u);
    }
  }
}
`;

export const SCATTER_DISTANCE_BINS_WGSL = /* wgsl */ `
${BINNING_COMMON_WGSL}

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> distanceOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> distanceCursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> distanceReferences: array<u32>;
@group(0) @binding(5) var<storage, read> params: CountParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.triangleBatchCount) {
    return;
  }
  let triangleIndex = params.triangleStart + globalId.x;
  if (triangleIndex >= params.triangleTotal) {
    return;
  }

  let indexOffset = triangleIndex * 3u;
  let a = loadPosition(indices[indexOffset]);
  let b = loadPosition(indices[indexOffset + 1u]);
  let c = loadPosition(indices[indexOffset + 2u]);
  let triangleMin = min(a, min(b, c));
  let triangleMax = max(a, max(b, c));
  if (!overlapsDomain(triangleMin, triangleMax)) {
    return;
  }

  let resolution = params.distanceBinResolution;
  let minimum = vec3<u32>(
    binCoordinate(triangleMin.x, params.domainMin.x, params.domainMax.x, resolution),
    binCoordinate(triangleMin.y, params.domainMin.y, params.domainMax.y, resolution),
    binCoordinate(triangleMin.z, params.domainMin.z, params.domainMax.z, resolution)
  );
  let maximum = vec3<u32>(
    binCoordinate(triangleMax.x, params.domainMin.x, params.domainMax.x, resolution),
    binCoordinate(triangleMax.y, params.domainMin.y, params.domainMax.y, resolution),
    binCoordinate(triangleMax.z, params.domainMin.z, params.domainMax.z, resolution)
  );
  for (var z = minimum.z; z <= maximum.z; z = z + 1u) {
    for (var y = minimum.y; y <= maximum.y; y = y + 1u) {
      for (var x = minimum.x; x <= maximum.x; x = x + 1u) {
        let binIndex = z * resolution * resolution + y * resolution + x;
        let cursor = atomicAdd(&distanceCursors[binIndex], 1u);
        distanceReferences[distanceOffsets[binIndex] + cursor] = triangleIndex;
      }
    }
  }
}
`;

export const SCATTER_PARITY_BINS_WGSL = /* wgsl */ `
${BINNING_COMMON_WGSL}

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> parityOffsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> parityCursors: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> parityReferences: array<u32>;
@group(0) @binding(5) var<storage, read> params: CountParams;

fn writeParityReference(binIndex: u32, triangleIndex: u32) {
  let cursor = atomicAdd(&parityCursors[binIndex], 1u);
  parityReferences[parityOffsets[binIndex] + cursor] = triangleIndex;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.triangleBatchCount) {
    return;
  }
  let triangleIndex = params.triangleStart + globalId.x;
  if (triangleIndex >= params.triangleTotal) {
    return;
  }

  let indexOffset = triangleIndex * 3u;
  let a = loadPosition(indices[indexOffset]);
  let b = loadPosition(indices[indexOffset + 1u]);
  let c = loadPosition(indices[indexOffset + 2u]);
  let triangleMin = min(a, min(b, c));
  let triangleMax = max(a, max(b, c));
  if (!overlapsDomain(triangleMin, triangleMax)) {
    return;
  }
  if (params.signMode != 1u && params.signMode != 3u) {
    return;
  }

  let resolution = params.parityBinResolution;
  let planeSize = resolution * resolution;
  let minX = binCoordinate(triangleMin.x, params.domainMin.x, params.domainMax.x, resolution);
  let minY = binCoordinate(triangleMin.y, params.domainMin.y, params.domainMax.y, resolution);
  let minZ = binCoordinate(triangleMin.z, params.domainMin.z, params.domainMax.z, resolution);
  let maxX = binCoordinate(triangleMax.x, params.domainMin.x, params.domainMax.x, resolution);
  let maxY = binCoordinate(triangleMax.y, params.domainMin.y, params.domainMax.y, resolution);
  let maxZ = binCoordinate(triangleMax.z, params.domainMin.z, params.domainMax.z, resolution);

  for (var z = minZ; z <= maxZ; z = z + 1u) {
    for (var y = minY; y <= maxY; y = y + 1u) {
      writeParityReference(z * resolution + y, triangleIndex);
    }
  }
  for (var z = minZ; z <= maxZ; z = z + 1u) {
    for (var x = minX; x <= maxX; x = x + 1u) {
      writeParityReference(planeSize + z * resolution + x, triangleIndex);
    }
  }
  for (var y = minY; y <= maxY; y = y + 1u) {
    for (var x = minX; x <= maxX; x = x + 1u) {
      writeParityReference(2u * planeSize + y * resolution + x, triangleIndex);
    }
  }
}
`;
