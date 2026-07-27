/*
 * Adapted from Vutify's mesh-SDF distance prototype and rewritten for:
 * vertex-sampled lattices, raw WebGPU, explicit shell mode, and three-axis
 * parity classification. No NVIDIA FlexiCubes source is present in this file.
 */

export const BAKE_DENSE_SDF_WGSL = /* wgsl */ `
struct BakeParams {
  sampleStart: u32,
  sampleBatchCount: u32,
  totalSampleCount: u32,
  triangleCount: u32,
  cellCountX: u32,
  cellCountY: u32,
  cellCountZ: u32,
  distanceBinResolution: u32,
  parityBinResolution: u32,
  signMode: u32,
  _padding0: u32,
  _padding1: u32,
  domainMin: vec4<f32>,
  domainMax: vec4<f32>,
  sampleSpacing: vec4<f32>,
  surface: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> distanceHeaders: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> distanceReferences: array<u32>;
@group(0) @binding(4) var<storage, read> parityHeaders: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> parityReferences: array<u32>;
@group(0) @binding(6) var<storage, read_write> sdfOutput: array<f32>;
@group(0) @binding(7) var<storage, read> params: BakeParams;

fn loadPosition(vertexIndex: u32) -> vec3<f32> {
  let offset = vertexIndex * 3u;
  return vec3<f32>(
    positions[offset],
    positions[offset + 1u],
    positions[offset + 2u]
  );
}

fn loadTriangle(triangleIndex: u32, corner: u32) -> vec3<f32> {
  return loadPosition(indices[triangleIndex * 3u + corner]);
}

fn pointSegmentDistanceSquared(
  point: vec3<f32>,
  start: vec3<f32>,
  end: vec3<f32>
) -> f32 {
  let edge = end - start;
  let edgeLengthSquared = dot(edge, edge);
  if (edgeLengthSquared == 0.0) {
    return dot(point - start, point - start);
  }
  let t = clamp(dot(point - start, edge) / edgeLengthSquared, 0.0, 1.0);
  let delta = point - (start + t * edge);
  return dot(delta, delta);
}

fn pointTriangleDistanceSquared(
  point: vec3<f32>,
  a: vec3<f32>,
  b: vec3<f32>,
  c: vec3<f32>
) -> f32 {
  let ab = b - a;
  let ac = c - a;
  let bc = c - b;
  let normal = cross(ab, ac);
  let maximumEdgeSquared = max(dot(ab, ab), max(dot(ac, ac), dot(bc, bc)));
  if (dot(normal, normal) <= maximumEdgeSquared * maximumEdgeSquared * 1e-12) {
    return min(
      pointSegmentDistanceSquared(point, a, b),
      min(
        pointSegmentDistanceSquared(point, b, c),
        pointSegmentDistanceSquared(point, c, a)
      )
    );
  }

  let ap = point - a;
  let d1 = dot(ab, ap);
  let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) {
    return dot(ap, ap);
  }

  let bp = point - b;
  let d3 = dot(ab, bp);
  let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) {
    return dot(bp, bp);
  }

  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
    let v = d1 / (d1 - d3);
    let delta = ap - v * ab;
    return dot(delta, delta);
  }

  let cp = point - c;
  let d5 = dot(ab, cp);
  let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) {
    return dot(cp, cp);
  }

  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
    let w = d2 / (d2 - d6);
    let delta = ap - w * ac;
    return dot(delta, delta);
  }

  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) {
    let w = (d4 - d3) / (d4 - d3 + d5 - d6);
    let delta = bp - w * bc;
    return dot(delta, delta);
  }

  let denominator = 1.0 / (va + vb + vc);
  let v = vb * denominator;
  let w = vc * denominator;
  let delta = ap - ab * v - ac * w;
  return dot(delta, delta);
}

fn aabbDistanceSquared(
  point: vec3<f32>,
  minimum: vec3<f32>,
  maximum: vec3<f32>
) -> f32 {
  let delta = max(max(minimum - point, vec3<f32>(0.0)), point - maximum);
  return dot(delta, delta);
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

fn hash32(input: u32) -> u32 {
  var value = input;
  value = value ^ (value >> 16u);
  value = value * 0x7feb352du;
  value = value ^ (value >> 15u);
  value = value * 0x846ca68bu;
  return value ^ (value >> 16u);
}

fn signedJitter(sampleIndex: u32, salt: u32) -> f32 {
  let bits = hash32(sampleIndex ^ salt) & 1023u;
  return f32(bits) / 1023.0 - 0.5;
}

fn rayTriangleDistance(
  origin: vec3<f32>,
  direction: vec3<f32>,
  a: vec3<f32>,
  b: vec3<f32>,
  c: vec3<f32>
) -> f32 {
  let edge1 = b - a;
  let edge2 = c - a;
  let p = cross(direction, edge2);
  let determinant = dot(edge1, p);
  if (abs(determinant) < 1e-8) {
    return -1.0;
  }

  let inverseDeterminant = 1.0 / determinant;
  let offset = origin - a;
  let u = dot(offset, p) * inverseDeterminant;
  let barycentricEpsilon = 1e-6;
  if (u < -barycentricEpsilon || u > 1.0 + barycentricEpsilon) {
    return -1.0;
  }

  let q = cross(offset, edge1);
  let v = dot(direction, q) * inverseDeterminant;
  if (v < -barycentricEpsilon || u + v > 1.0 + barycentricEpsilon) {
    return -1.0;
  }

  let distance = dot(edge2, q) * inverseDeterminant;
  return select(-1.0, distance, distance > params.surface.y);
}

fn projectionHeaderIndex(axis: u32, point: vec3<f32>) -> u32 {
  let resolution = params.parityBinResolution;
  let planeSize = resolution * resolution;
  if (axis == 0u) {
    let u = binCoordinate(point.y, params.domainMin.y, params.domainMax.y, resolution);
    let v = binCoordinate(point.z, params.domainMin.z, params.domainMax.z, resolution);
    return v * resolution + u;
  }
  if (axis == 1u) {
    let u = binCoordinate(point.x, params.domainMin.x, params.domainMax.x, resolution);
    let v = binCoordinate(point.z, params.domainMin.z, params.domainMax.z, resolution);
    return planeSize + v * resolution + u;
  }
  let u = binCoordinate(point.x, params.domainMin.x, params.domainMax.x, resolution);
  let v = binCoordinate(point.y, params.domainMin.y, params.domainMax.y, resolution);
  return 2u * planeSize + v * resolution + u;
}

fn classifyAxisParity(
  point: vec3<f32>,
  sampleIndex: u32,
  axis: u32
) -> bool {
  let jitterScale = max(params.surface.y * 8.0, 1e-7);
  var origin = point;
  var direction = vec3<f32>(0.0);
  if (axis == 0u) {
    origin.y = origin.y + signedJitter(sampleIndex, 0x68bc21ebu) * jitterScale;
    origin.z = origin.z + signedJitter(sampleIndex, 0x02e5be93u) * jitterScale;
    direction.x = 1.0;
  } else if (axis == 1u) {
    origin.x = origin.x + signedJitter(sampleIndex, 0x967a889bu) * jitterScale;
    origin.z = origin.z + signedJitter(sampleIndex, 0x4f1bbcdcu) * jitterScale;
    direction.y = 1.0;
  } else {
    origin.x = origin.x + signedJitter(sampleIndex, 0x7a2d9b61u) * jitterScale;
    origin.y = origin.y + signedJitter(sampleIndex, 0xc3a5c85cu) * jitterScale;
    direction.z = 1.0;
  }

  let header = parityHeaders[projectionHeaderIndex(axis, origin)];
  var intersectionCount = 0u;
  for (var offset = 0u; offset < header.y; offset = offset + 1u) {
    let triangleIndex = parityReferences[header.x + offset];
    let a = loadTriangle(triangleIndex, 0u);
    let b = loadTriangle(triangleIndex, 1u);
    let c = loadTriangle(triangleIndex, 2u);
    if (rayTriangleDistance(origin, direction, a, b, c) > 0.0) {
      intersectionCount = intersectionCount + 1u;
    }
  }
  return (intersectionCount & 1u) == 1u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= params.sampleBatchCount) {
    return;
  }
  let sampleIndex = params.sampleStart + globalId.x;
  if (sampleIndex >= params.totalSampleCount) {
    return;
  }

  let sampleCountX = params.cellCountX + 1u;
  let sampleCountY = params.cellCountY + 1u;
  let x = sampleIndex % sampleCountX;
  let y = (sampleIndex / sampleCountX) % sampleCountY;
  let z = sampleIndex / (sampleCountX * sampleCountY);
  let point = params.domainMin.xyz + vec3<f32>(f32(x), f32(y), f32(z)) * params.sampleSpacing.xyz;

  var minimumDistanceSquared = 3.402823e38;
  let resolution = params.distanceBinResolution;
  let binSize = (params.domainMax.xyz - params.domainMin.xyz) / f32(resolution);
  let centerBin = vec3<i32>(
    i32(binCoordinate(point.x, params.domainMin.x, params.domainMax.x, resolution)),
    i32(binCoordinate(point.y, params.domainMin.y, params.domainMax.y, resolution)),
    i32(binCoordinate(point.z, params.domainMin.z, params.domainMax.z, resolution))
  );
  let maximumBin = i32(resolution) - 1;
  for (var radius = 0; radius < i32(resolution); radius = radius + 1) {
    let minimum = max(centerBin - vec3<i32>(radius), vec3<i32>(0));
    let maximum = min(centerBin + vec3<i32>(radius), vec3<i32>(maximumBin));
    for (var z = minimum.z; z <= maximum.z; z = z + 1) {
      for (var y = minimum.y; y <= maximum.y; y = y + 1) {
        for (var x = minimum.x; x <= maximum.x; x = x + 1) {
          let shellDistance = max(
            abs(x - centerBin.x),
            max(abs(y - centerBin.y), abs(z - centerBin.z))
          );
          if (shellDistance != radius) {
            continue;
          }
          let bx = u32(x);
          let by = u32(y);
          let bz = u32(z);
          let binIndex = bz * resolution * resolution + by * resolution + bx;
          let binMinimum = params.domainMin.xyz + vec3<f32>(
            f32(bx),
            f32(by),
            f32(bz)
          ) * binSize;
          let binMaximum = binMinimum + binSize;
          if (aabbDistanceSquared(point, binMinimum, binMaximum) > minimumDistanceSquared) {
            continue;
          }

          let header = distanceHeaders[binIndex];
          for (var offset = 0u; offset < header.y; offset = offset + 1u) {
            let triangleIndex = distanceReferences[header.x + offset];
            let a = loadTriangle(triangleIndex, 0u);
            let b = loadTriangle(triangleIndex, 1u);
            let c = loadTriangle(triangleIndex, 2u);
            minimumDistanceSquared = min(
              minimumDistanceSquared,
              pointTriangleDistanceSquared(point, a, b, c)
            );
          }
        }
      }
    }

    var nearestUnscannedDistance = 3.402823e38;
    let scannedMinimum = params.domainMin.xyz + vec3<f32>(minimum) * binSize;
    let scannedMaximum = params.domainMin.xyz + vec3<f32>(maximum + vec3<i32>(1)) * binSize;
    if (minimum.x > 0) {
      nearestUnscannedDistance = min(nearestUnscannedDistance, point.x - scannedMinimum.x);
    }
    if (minimum.y > 0) {
      nearestUnscannedDistance = min(nearestUnscannedDistance, point.y - scannedMinimum.y);
    }
    if (minimum.z > 0) {
      nearestUnscannedDistance = min(nearestUnscannedDistance, point.z - scannedMinimum.z);
    }
    if (maximum.x < maximumBin) {
      nearestUnscannedDistance = min(nearestUnscannedDistance, scannedMaximum.x - point.x);
    }
    if (maximum.y < maximumBin) {
      nearestUnscannedDistance = min(nearestUnscannedDistance, scannedMaximum.y - point.y);
    }
    if (maximum.z < maximumBin) {
      nearestUnscannedDistance = min(nearestUnscannedDistance, scannedMaximum.z - point.z);
    }
    if (
      nearestUnscannedDistance * nearestUnscannedDistance
      > minimumDistanceSquared
    ) {
      break;
    }
  }

  let unsignedDistance = sqrt(minimumDistanceSquared);
  var value = unsignedDistance;
  if (params.signMode == 2u) {
    value = unsignedDistance - params.surface.x;
  } else if (
    (params.signMode == 1u || params.signMode == 3u)
    && unsignedDistance > params.surface.y
  ) {
    var insideVotes = 0u;
    if (classifyAxisParity(point, sampleIndex, 0u)) {
      insideVotes = insideVotes + 1u;
    }
    if (classifyAxisParity(point, sampleIndex, 1u)) {
      insideVotes = insideVotes + 1u;
    }
    if (classifyAxisParity(point, sampleIndex, 2u)) {
      insideVotes = insideVotes + 1u;
    }
    if (insideVotes >= 2u) {
      value = -unsignedDistance;
    }
  } else if (unsignedDistance <= params.surface.y) {
    value = 0.0;
  }
  if (params.signMode == 3u) {
    value = min(value, unsignedDistance - params.surface.x);
  }
  sdfOutput[sampleIndex] = value;
}
`;
