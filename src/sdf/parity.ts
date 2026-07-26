import type { Vec3 } from "../core/types.js";
import type { PreparedTriangle } from "./triangle.js";

const RAY_DIRECTIONS: readonly Vec3[] = [
  [0.923364, 0.342151, 0.174781],
  [-0.204873, 0.858143, 0.470825],
  [0.361744, -0.268118, 0.892929],
];

function rayTriangleIntersection(
  origin: Vec3,
  direction: Vec3,
  triangle: PreparedTriangle,
): number | undefined {
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

  if (Math.abs(determinant) < 1e-12) {
    return undefined;
  }

  const inverseDeterminant = 1 / determinant;
  const offsetX = origin[0] - triangle.a[0];
  const offsetY = origin[1] - triangle.a[1];
  const offsetZ = origin[2] - triangle.a[2];
  const u = (
    offsetX * pX
    + offsetY * pY
    + offsetZ * pZ
  ) * inverseDeterminant;
  const barycentricEpsilon = 1e-10;
  if (u < -barycentricEpsilon || u > 1 + barycentricEpsilon) {
    return undefined;
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
    return undefined;
  }

  const distance = (
    edge2X * qX
    + edge2Y * qY
    + edge2Z * qZ
  ) * inverseDeterminant;
  return distance > 1e-10 ? distance : undefined;
}

function countUniqueIntersections(
  point: Vec3,
  direction: Vec3,
  triangles: readonly PreparedTriangle[],
): number {
  const intersections: number[] = [];
  for (const triangle of triangles) {
    const intersection = rayTriangleIntersection(point, direction, triangle);
    if (intersection !== undefined) {
      intersections.push(intersection);
    }
  }

  intersections.sort((left, right) => left - right);
  let uniqueCount = 0;
  let previous = -Infinity;
  for (const intersection of intersections) {
    const tolerance = 1e-8 * Math.max(1, Math.abs(intersection));
    if (intersection - previous > tolerance) {
      uniqueCount++;
      previous = intersection;
    }
  }
  return uniqueCount;
}

export interface ParityClassification {
  readonly inside: boolean;
  readonly rayTriangleTests: number;
}

export function classifyPointByParity(
  point: Vec3,
  triangles: readonly PreparedTriangle[],
): ParityClassification {
  let insideVotes = 0;
  for (const direction of RAY_DIRECTIONS) {
    if (countUniqueIntersections(point, direction, triangles) % 2 === 1) {
      insideVotes++;
    }
  }

  return {
    inside: insideVotes >= 2,
    rayTriangleTests: RAY_DIRECTIONS.length * triangles.length,
  };
}
