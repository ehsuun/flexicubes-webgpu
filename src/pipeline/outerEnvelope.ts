import type {
  ExtractedMesh,
  IndexedMesh,
  ProxyOuterEnvelopeEvidence,
  ProxyOuterEnvelopeVerificationEvidence,
  ProxyOuterEnvelopeVerificationOptions,
} from "../core/types.js";
import { validateIndexedMesh } from "../core/mesh.js";

const LEAF_TRIANGLE_COUNT = 8;
const DEFAULT_MAXIMUM_SOURCE_SAMPLES = 8_192;
const DEFAULT_SAMPLE_BATCH_SIZE = 512;

type BvhNode = {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
  left: number;
  right: number;
  first: number;
  count: number;
};

type ProxyBvh = Readonly<{
  positions: Float32Array;
  indices: Uint32Array;
  triangleOrder: Uint32Array;
  triangleNormals: Float32Array;
  nodes: readonly BvhNode[];
}>;

type QueryScratch = {
  stack: Int32Array;
  barycentrics: Float64Array;
  triangleIndex: number;
  distanceSquared: number;
  barycentricU: number;
  barycentricV: number;
  barycentricW: number;
};

type VerificationProgress = (
  completedSamples: number,
  totalSamples: number,
) => void;

export class ProxyOuterEnvelopeError extends Error {
  public override readonly name = "ProxyOuterEnvelopeError";

  public constructor(
    message: string,
    public readonly evidence: ProxyOuterEnvelopeEvidence,
  ) {
    super(message);
  }
}

const throwIfAborted = (signal: AbortSignal | undefined): void => {
  if (signal?.aborted !== true) return;
  const error = new Error("proxy outer-envelope verification was aborted");
  error.name = "AbortError";
  throw error;
};

const positiveInteger = (
  name: string,
  value: number | undefined,
  fallback: number,
): number => {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < 1) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
  return resolved;
};

const validateMinimumSeparation = (value: number): void => {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(
      "minimumSeparation must be finite and non-negative",
    );
  }
};

const yieldToMainThread = (): Promise<void> => new Promise(resolve => {
  setTimeout(resolve, 0);
});

const coordinate = (
  positions: Float32Array,
  vertexIndex: number,
  axis: 0 | 1 | 2,
): number => positions[vertexIndex * 3 + axis]!;

const triangleBounds = (
  positions: Float32Array,
  indices: Uint32Array,
  triangleIndex: number,
): readonly [number, number, number, number, number, number] => {
  const first = indices[triangleIndex * 3]!;
  const second = indices[triangleIndex * 3 + 1]!;
  const third = indices[triangleIndex * 3 + 2]!;
  const ax = coordinate(positions, first, 0);
  const ay = coordinate(positions, first, 1);
  const az = coordinate(positions, first, 2);
  const bx = coordinate(positions, second, 0);
  const by = coordinate(positions, second, 1);
  const bz = coordinate(positions, second, 2);
  const cx = coordinate(positions, third, 0);
  const cy = coordinate(positions, third, 1);
  const cz = coordinate(positions, third, 2);
  return [
    Math.min(ax, bx, cx),
    Math.min(ay, by, cy),
    Math.min(az, bz, cz),
    Math.max(ax, bx, cx),
    Math.max(ay, by, cy),
    Math.max(az, bz, cz),
  ];
};

const normalizedTriangleNormal = (
  positions: Float32Array,
  indices: Uint32Array,
  triangleIndex: number,
): readonly [number, number, number] => {
  const first = indices[triangleIndex * 3]!;
  const second = indices[triangleIndex * 3 + 1]!;
  const third = indices[triangleIndex * 3 + 2]!;
  const abx = coordinate(positions, second, 0)
    - coordinate(positions, first, 0);
  const aby = coordinate(positions, second, 1)
    - coordinate(positions, first, 1);
  const abz = coordinate(positions, second, 2)
    - coordinate(positions, first, 2);
  const acx = coordinate(positions, third, 0)
    - coordinate(positions, first, 0);
  const acy = coordinate(positions, third, 1)
    - coordinate(positions, first, 1);
  const acz = coordinate(positions, third, 2)
    - coordinate(positions, first, 2);
  const x = aby * acz - abz * acy;
  const y = abz * acx - abx * acz;
  const z = abx * acy - aby * acx;
  const length = Math.hypot(x, y, z);
  if (length === 0) {
    throw new RangeError("proxy outer-envelope input has a degenerate triangle");
  }
  return [x / length, y / length, z / length];
};

const buildProxyBvh = (mesh: ExtractedMesh): ProxyBvh => {
  validateIndexedMesh(mesh);
  if (mesh.sourceCells.length !== mesh.positions.length / 3) {
    throw new RangeError("proxy sourceCells must match its vertex count");
  }
  const triangleCount = mesh.indices.length / 3;
  const order = Array.from(
    { length: triangleCount },
    (_, triangleIndex) => triangleIndex,
  );
  const bounds = order.map(triangleIndex => triangleBounds(
    mesh.positions,
    mesh.indices,
    triangleIndex,
  ));
  const normals = new Float32Array(triangleCount * 3);
  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
    normals.set(
      normalizedTriangleNormal(mesh.positions, mesh.indices, triangleIndex),
      triangleIndex * 3,
    );
  }
  const nodes: BvhNode[] = [];
  const buildNode = (first: number, count: number): number => {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let offset = first; offset < first + count; offset++) {
      const triangle = bounds[order[offset]!]!;
      minX = Math.min(minX, triangle[0]);
      minY = Math.min(minY, triangle[1]);
      minZ = Math.min(minZ, triangle[2]);
      maxX = Math.max(maxX, triangle[3]);
      maxY = Math.max(maxY, triangle[4]);
      maxZ = Math.max(maxZ, triangle[5]);
    }
    const nodeIndex = nodes.length;
    nodes.push({
      minX,
      minY,
      minZ,
      maxX,
      maxY,
      maxZ,
      left: -1,
      right: -1,
      first,
      count,
    });
    if (count <= LEAF_TRIANGLE_COUNT) return nodeIndex;
    const extentX = maxX - minX;
    const extentY = maxY - minY;
    const extentZ = maxZ - minZ;
    const axis = extentY > extentX
      ? extentZ > extentY ? 2 : 1
      : extentZ > extentX ? 2 : 0;
    const sorted = order.slice(first, first + count).sort((left, right) => (
      bounds[left]![axis]! + bounds[left]![axis + 3]!
      - bounds[right]![axis]! - bounds[right]![axis + 3]!
    ));
    for (let offset = 0; offset < sorted.length; offset++) {
      order[first + offset] = sorted[offset]!;
    }
    const leftCount = Math.floor(count / 2);
    nodes[nodeIndex]!.left = buildNode(first, leftCount);
    nodes[nodeIndex]!.right = buildNode(
      first + leftCount,
      count - leftCount,
    );
    return nodeIndex;
  };
  buildNode(0, triangleCount);
  return {
    positions: mesh.positions,
    indices: mesh.indices,
    triangleOrder: Uint32Array.from(order),
    triangleNormals: normals,
    nodes,
  };
};

const aabbDistanceSquared = (
  node: BvhNode,
  x: number,
  y: number,
  z: number,
): number => {
  const dx = Math.max(node.minX - x, 0, x - node.maxX);
  const dy = Math.max(node.minY - y, 0, y - node.maxY);
  const dz = Math.max(node.minZ - z, 0, z - node.maxZ);
  return dx * dx + dy * dy + dz * dz;
};

const closestPointDistanceSquared = (
  bvh: ProxyBvh,
  triangleIndex: number,
  px: number,
  py: number,
  pz: number,
  barycentrics: Float64Array,
): number => {
  const first = bvh.indices[triangleIndex * 3]!;
  const second = bvh.indices[triangleIndex * 3 + 1]!;
  const third = bvh.indices[triangleIndex * 3 + 2]!;
  const ax = coordinate(bvh.positions, first, 0);
  const ay = coordinate(bvh.positions, first, 1);
  const az = coordinate(bvh.positions, first, 2);
  const abx = coordinate(bvh.positions, second, 0) - ax;
  const aby = coordinate(bvh.positions, second, 1) - ay;
  const abz = coordinate(bvh.positions, second, 2) - az;
  const acx = coordinate(bvh.positions, third, 0) - ax;
  const acy = coordinate(bvh.positions, third, 1) - ay;
  const acz = coordinate(bvh.positions, third, 2) - az;
  const apx = px - ax;
  const apy = py - ay;
  const apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  let u = 0;
  let v = 0;
  let w = 0;
  if (d1 <= 0 && d2 <= 0) {
    u = 1;
  } else {
    const bpx = apx - abx;
    const bpy = apy - aby;
    const bpz = apz - abz;
    const d3 = abx * bpx + aby * bpy + abz * bpz;
    const d4 = acx * bpx + acy * bpy + acz * bpz;
    if (d3 >= 0 && d4 <= d3) {
      v = 1;
    } else {
      const vc = d1 * d4 - d3 * d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        v = d1 / (d1 - d3);
        u = 1 - v;
      } else {
        const cpx = apx - acx;
        const cpy = apy - acy;
        const cpz = apz - acz;
        const d5 = abx * cpx + aby * cpy + abz * cpz;
        const d6 = acx * cpx + acy * cpy + acz * cpz;
        if (d6 >= 0 && d5 <= d6) {
          w = 1;
        } else {
          const vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            w = d2 / (d2 - d6);
            u = 1 - w;
          } else {
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
              w = (d4 - d3) / (d4 - d3 + d5 - d6);
              v = 1 - w;
            } else {
              const inverse = 1 / (va + vb + vc);
              v = vb * inverse;
              w = vc * inverse;
              u = 1 - v - w;
            }
          }
        }
      }
    }
  }
  barycentrics[0] = u;
  barycentrics[1] = v;
  barycentrics[2] = w;
  const dx = apx - abx * v - acx * w;
  const dy = apy - aby * v - acy * w;
  const dz = apz - abz * v - acz * w;
  return dx * dx + dy * dy + dz * dz;
};

const createQueryScratch = (bvh: ProxyBvh): QueryScratch => ({
  stack: new Int32Array(bvh.nodes.length),
  barycentrics: new Float64Array(3),
  triangleIndex: 0,
  distanceSquared: Infinity,
  barycentricU: 0,
  barycentricV: 0,
  barycentricW: 0,
});

const queryClosestTriangle = (
  bvh: ProxyBvh,
  scratch: QueryScratch,
  x: number,
  y: number,
  z: number,
): void => {
  let bestTriangle = 0;
  let bestDistanceSquared = Infinity;
  let bestU = 0;
  let bestV = 0;
  let bestW = 0;
  let stackLength = 1;
  scratch.stack[0] = 0;
  while (stackLength > 0) {
    const node = bvh.nodes[scratch.stack[--stackLength]!]!;
    if (aabbDistanceSquared(node, x, y, z) > bestDistanceSquared) continue;
    if (node.left >= 0 && node.right >= 0) {
      const leftDistance = aabbDistanceSquared(
        bvh.nodes[node.left]!, x, y, z,
      );
      const rightDistance = aabbDistanceSquared(
        bvh.nodes[node.right]!, x, y, z,
      );
      scratch.stack[stackLength++] = leftDistance < rightDistance
        ? node.right
        : node.left;
      scratch.stack[stackLength++] = leftDistance < rightDistance
        ? node.left
        : node.right;
      continue;
    }
    for (let offset = node.first; offset < node.first + node.count; offset++) {
      const triangleIndex = bvh.triangleOrder[offset]!;
      const distanceSquared = closestPointDistanceSquared(
        bvh,
        triangleIndex,
        x,
        y,
        z,
        scratch.barycentrics,
      );
      if (distanceSquared >= bestDistanceSquared) continue;
      bestTriangle = triangleIndex;
      bestDistanceSquared = distanceSquared;
      bestU = scratch.barycentrics[0]!;
      bestV = scratch.barycentrics[1]!;
      bestW = scratch.barycentrics[2]!;
    }
  }
  scratch.triangleIndex = bestTriangle;
  scratch.distanceSquared = bestDistanceSquared;
  scratch.barycentricU = bestU;
  scratch.barycentricV = bestV;
  scratch.barycentricW = bestW;
};

const evenlySpacedIndex = (
  ordinal: number,
  selectedCount: number,
  candidateCount: number,
): number => Math.floor(
  (ordinal + 0.5) * candidateCount / selectedCount,
);

const createSourceSamples = (
  mesh: IndexedMesh,
  maximumSamples: number,
): Float64Array => {
  validateIndexedMesh(mesh);
  const vertexCount = mesh.positions.length / 3;
  const triangleCount = mesh.indices.length / 3;
  const candidateCount = vertexCount + triangleCount;
  const sampleCount = Math.min(maximumSamples, candidateCount);
  let vertexSampleCount = Math.min(
    vertexCount,
    Math.ceil(sampleCount / 2),
  );
  let triangleSampleCount = Math.min(
    triangleCount,
    sampleCount - vertexSampleCount,
  );
  const remaining = sampleCount - vertexSampleCount - triangleSampleCount;
  vertexSampleCount += Math.min(remaining, vertexCount - vertexSampleCount);
  triangleSampleCount = sampleCount - vertexSampleCount;
  const samples = new Float64Array(sampleCount * 3);
  for (let ordinal = 0; ordinal < vertexSampleCount; ordinal++) {
    const vertexIndex = evenlySpacedIndex(
      ordinal,
      vertexSampleCount,
      vertexCount,
    );
    samples.set(mesh.positions.subarray(vertexIndex * 3, vertexIndex * 3 + 3),
      ordinal * 3);
  }
  for (let ordinal = 0; ordinal < triangleSampleCount; ordinal++) {
    const triangleIndex = evenlySpacedIndex(
      ordinal,
      triangleSampleCount,
      triangleCount,
    );
    const output = (vertexSampleCount + ordinal) * 3;
    let x = 0;
    let y = 0;
    let z = 0;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = mesh.indices[triangleIndex * 3 + corner]!;
      x += mesh.positions[vertex * 3]! / 3;
      y += mesh.positions[vertex * 3 + 1]! / 3;
      z += mesh.positions[vertex * 3 + 2]! / 3;
    }
    samples.set([x, y, z], output);
  }
  return samples;
};

const signedSeparation = (
  bvh: ProxyBvh,
  scratch: QueryScratch,
  x: number,
  y: number,
  z: number,
): number => {
  queryClosestTriangle(bvh, scratch, x, y, z);
  const triangleOffset = scratch.triangleIndex * 3;
  const first = bvh.indices[triangleOffset]!;
  const second = bvh.indices[triangleOffset + 1]!;
  const third = bvh.indices[triangleOffset + 2]!;
  const closestX = coordinate(bvh.positions, first, 0) * scratch.barycentricU
    + coordinate(bvh.positions, second, 0) * scratch.barycentricV
    + coordinate(bvh.positions, third, 0) * scratch.barycentricW;
  const closestY = coordinate(bvh.positions, first, 1) * scratch.barycentricU
    + coordinate(bvh.positions, second, 1) * scratch.barycentricV
    + coordinate(bvh.positions, third, 1) * scratch.barycentricW;
  const closestZ = coordinate(bvh.positions, first, 2) * scratch.barycentricU
    + coordinate(bvh.positions, second, 2) * scratch.barycentricV
    + coordinate(bvh.positions, third, 2) * scratch.barycentricW;
  return (
    (closestX - x) * bvh.triangleNormals[triangleOffset]!
    + (closestY - y) * bvh.triangleNormals[triangleOffset + 1]!
    + (closestZ - z) * bvh.triangleNormals[triangleOffset + 2]!
  );
};

const verify = async (
  source: IndexedMesh,
  proxy: ExtractedMesh,
  options: ProxyOuterEnvelopeVerificationOptions,
  progress?: VerificationProgress,
): Promise<ProxyOuterEnvelopeVerificationEvidence> => {
  validateMinimumSeparation(options.minimumSeparation);
  const maximumSamples = positiveInteger(
    "maximumSourceSamples",
    options.maximumSourceSamples,
    DEFAULT_MAXIMUM_SOURCE_SAMPLES,
  );
  const batchSize = positiveInteger(
    "sampleBatchSize",
    options.sampleBatchSize,
    DEFAULT_SAMPLE_BATCH_SIZE,
  );
  throwIfAborted(options.signal);
  const samples = createSourceSamples(source, maximumSamples);
  const bvh = buildProxyBvh(proxy);
  const scratch = createQueryScratch(bvh);
  const sampleCount = samples.length / 3;
  let violationCount = 0;
  let minimumSignedSeparation = Infinity;
  for (let start = 0; start < sampleCount; start += batchSize) {
    throwIfAborted(options.signal);
    const end = Math.min(sampleCount, start + batchSize);
    for (let sampleIndex = start; sampleIndex < end; sampleIndex++) {
      const offset = sampleIndex * 3;
      const separation = signedSeparation(
        bvh,
        scratch,
        samples[offset]!,
        samples[offset + 1]!,
        samples[offset + 2]!,
      );
      minimumSignedSeparation = Math.min(
        minimumSignedSeparation,
        separation,
      );
      if (separation < options.minimumSeparation) violationCount++;
    }
    progress?.(end, sampleCount);
    if (end < sampleCount) await yieldToMainThread();
  }
  return {
    method: "sampled-source-surface",
    sampleStrategy: "uniform-vertices-and-triangle-centroids",
    sourceSampleCount: sampleCount,
    violationCount,
    minimumSignedSeparation,
    maximumIngress: Math.max(
      0,
      options.minimumSeparation - minimumSignedSeparation,
    ),
    queryCount: sampleCount,
  };
};

export const verifyProxyOuterEnvelope = (
  source: IndexedMesh,
  proxy: ExtractedMesh,
  options: ProxyOuterEnvelopeVerificationOptions,
): Promise<ProxyOuterEnvelopeVerificationEvidence> => verify(
  source,
  proxy,
  options,
);

export const verifyProxyOuterEnvelopeWithProgress = verify;
