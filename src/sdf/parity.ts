import type { Vec3 } from "../core/types.js";
import type { PreparedTriangle } from "./triangle.js";

const AXIS_DIRECTIONS: readonly Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

const JITTER_SALTS = [
  [0x68bc21eb, 0x02e5be93],
  [0x967a889b, 0x4f1bbcdc],
  [0x7a2d9b61, 0xc3a5c85c],
] as const;
const JITTER_AXES = [
  [1, 2],
  [0, 2],
  [0, 1],
] as const;

function hash32(input: number): number {
  let value = input >>> 0;
  value = (value ^ (value >>> 16)) >>> 0;
  value = Math.imul(value, 0x7feb352d) >>> 0;
  value = (value ^ (value >>> 15)) >>> 0;
  value = Math.imul(value, 0x846ca68b) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

function signedJitter(sampleIndex: number, salt: number): number {
  return (hash32((sampleIndex ^ salt) >>> 0) & 1023) / 1023 - 0.5;
}

function jitteredOrigin(
  point: Vec3,
  sampleIndex: number,
  axis: number,
  jitterScale: number,
): Vec3 {
  const origin: [number, number, number] = [
    point[0],
    point[1],
    point[2],
  ];
  const axes = JITTER_AXES[axis]!;
  const salts = JITTER_SALTS[axis]!;
  origin[axes[0]] += signedJitter(sampleIndex, salts[0]) * jitterScale;
  origin[axes[1]] += signedJitter(sampleIndex, salts[1]) * jitterScale;
  return origin;
}

function rayTriangleIntersection(
  origin: Vec3,
  direction: Vec3,
  triangle: PreparedTriangle,
  surfaceEpsilon: number,
): boolean {
  const edge1X = triangle.b[0] - triangle.a[0];
  const edge1Y = triangle.b[1] - triangle.a[1];
  const edge1Z = triangle.b[2] - triangle.a[2];
  const edge2X = triangle.c[0] - triangle.a[0];
  const edge2Y = triangle.c[1] - triangle.a[1];
  const edge2Z = triangle.c[2] - triangle.a[2];
  const pX = direction[1] * edge2Z - direction[2] * edge2Y;
  const pY = direction[2] * edge2X - direction[0] * edge2Z;
  const pZ = direction[0] * edge2Y - direction[1] * edge2X;
  const determinant = edge1X * pX + edge1Y * pY + edge1Z * pZ;
  if (Math.abs(determinant) < 1e-8) return false;

  const inverseDeterminant = 1 / determinant;
  const offsetX = origin[0] - triangle.a[0];
  const offsetY = origin[1] - triangle.a[1];
  const offsetZ = origin[2] - triangle.a[2];
  const u = (
    offsetX * pX
    + offsetY * pY
    + offsetZ * pZ
  ) * inverseDeterminant;
  const barycentricEpsilon = 1e-6;
  if (u < -barycentricEpsilon || u > 1 + barycentricEpsilon) {
    return false;
  }

  const qX = offsetY * edge1Z - offsetZ * edge1Y;
  const qY = offsetZ * edge1X - offsetX * edge1Z;
  const qZ = offsetX * edge1Y - offsetY * edge1X;
  const v = (
    direction[0] * qX
    + direction[1] * qY
    + direction[2] * qZ
  ) * inverseDeterminant;
  if (
    v < -barycentricEpsilon
    || u + v > 1 + barycentricEpsilon
  ) {
    return false;
  }

  const distance = (
    edge2X * qX
    + edge2Y * qY
    + edge2Z * qZ
  ) * inverseDeterminant;
  return distance > surfaceEpsilon;
}

export interface ParityClassification {
  readonly inside: boolean;
  readonly rayTriangleTests: number;
}

export function classifyPointByParity(
  point: Vec3,
  triangles: readonly PreparedTriangle[],
  sampleIndex: number,
  surfaceEpsilon: number,
): ParityClassification {
  const jitterScale = Math.max(surfaceEpsilon * 8, 1e-7);
  let insideVotes = 0;
  for (let axis = 0; axis < AXIS_DIRECTIONS.length; axis++) {
    const origin = jitteredOrigin(
      point,
      sampleIndex,
      axis,
      jitterScale,
    );
    let intersectionCount = 0;
    for (const triangle of triangles) {
      intersectionCount += Number(rayTriangleIntersection(
        origin,
        AXIS_DIRECTIONS[axis]!,
        triangle,
        surfaceEpsilon,
      ));
    }
    insideVotes += Number((intersectionCount & 1) === 1);
  }
  return {
    inside: insideVotes >= 2,
    rayTriangleTests: AXIS_DIRECTIONS.length * triangles.length,
  };
}
