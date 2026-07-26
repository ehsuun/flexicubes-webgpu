import type { IndexedMesh, Vec3 } from "../core/types.js";
import { validateIndexedMesh } from "../core/mesh.js";

export interface PreparedTriangle {
  readonly a: Vec3;
  readonly b: Vec3;
  readonly c: Vec3;
}

function squaredLength(x: number, y: number, z: number): number {
  return x * x + y * y + z * z;
}

function pointSegmentDistanceSquared(
  point: Vec3,
  start: Vec3,
  end: Vec3,
): number {
  const edgeX = end[0] - start[0];
  const edgeY = end[1] - start[1];
  const edgeZ = end[2] - start[2];
  const pointX = point[0] - start[0];
  const pointY = point[1] - start[1];
  const pointZ = point[2] - start[2];
  const edgeLengthSquared = squaredLength(edgeX, edgeY, edgeZ);

  if (edgeLengthSquared === 0) {
    return squaredLength(pointX, pointY, pointZ);
  }

  const projection = (
    pointX * edgeX
    + pointY * edgeY
    + pointZ * edgeZ
  ) / edgeLengthSquared;
  const t = Math.max(0, Math.min(1, projection));
  return squaredLength(
    pointX - t * edgeX,
    pointY - t * edgeY,
    pointZ - t * edgeZ,
  );
}

function degenerateTriangleDistanceSquared(
  point: Vec3,
  triangle: PreparedTriangle,
): number {
  return Math.min(
    pointSegmentDistanceSquared(point, triangle.a, triangle.b),
    pointSegmentDistanceSquared(point, triangle.b, triangle.c),
    pointSegmentDistanceSquared(point, triangle.c, triangle.a),
  );
}

export function pointTriangleDistanceSquared(
  point: Vec3,
  triangle: PreparedTriangle,
): number {
  const abX = triangle.b[0] - triangle.a[0];
  const abY = triangle.b[1] - triangle.a[1];
  const abZ = triangle.b[2] - triangle.a[2];
  const acX = triangle.c[0] - triangle.a[0];
  const acY = triangle.c[1] - triangle.a[1];
  const acZ = triangle.c[2] - triangle.a[2];
  const normalX = abY * acZ - abZ * acY;
  const normalY = abZ * acX - abX * acZ;
  const normalZ = abX * acY - abY * acX;
  const maximumEdgeSquared = Math.max(
    squaredLength(abX, abY, abZ),
    squaredLength(acX, acY, acZ),
    squaredLength(
      triangle.c[0] - triangle.b[0],
      triangle.c[1] - triangle.b[1],
      triangle.c[2] - triangle.b[2],
    ),
  );
  const areaSquared = squaredLength(normalX, normalY, normalZ);
  if (areaSquared <= maximumEdgeSquared * maximumEdgeSquared * 1e-24) {
    return degenerateTriangleDistanceSquared(point, triangle);
  }

  const apX = point[0] - triangle.a[0];
  const apY = point[1] - triangle.a[1];
  const apZ = point[2] - triangle.a[2];
  const d1 = abX * apX + abY * apY + abZ * apZ;
  const d2 = acX * apX + acY * apY + acZ * apZ;
  if (d1 <= 0 && d2 <= 0) {
    return squaredLength(apX, apY, apZ);
  }

  const bpX = point[0] - triangle.b[0];
  const bpY = point[1] - triangle.b[1];
  const bpZ = point[2] - triangle.b[2];
  const d3 = abX * bpX + abY * bpY + abZ * bpZ;
  const d4 = acX * bpX + acY * bpY + acZ * bpZ;
  if (d3 >= 0 && d4 <= d3) {
    return squaredLength(bpX, bpY, bpZ);
  }

  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return squaredLength(
      apX - v * abX,
      apY - v * abY,
      apZ - v * abZ,
    );
  }

  const cpX = point[0] - triangle.c[0];
  const cpY = point[1] - triangle.c[1];
  const cpZ = point[2] - triangle.c[2];
  const d5 = abX * cpX + abY * cpY + abZ * cpZ;
  const d6 = acX * cpX + acY * cpY + acZ * cpZ;
  if (d6 >= 0 && d5 <= d6) {
    return squaredLength(cpX, cpY, cpZ);
  }

  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return squaredLength(
      apX - w * acX,
      apY - w * acY,
      apZ - w * acZ,
    );
  }

  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const edgeX = triangle.c[0] - triangle.b[0];
    const edgeY = triangle.c[1] - triangle.b[1];
    const edgeZ = triangle.c[2] - triangle.b[2];
    const w = (d4 - d3) / (d4 - d3 + d5 - d6);
    return squaredLength(
      bpX - w * edgeX,
      bpY - w * edgeY,
      bpZ - w * edgeZ,
    );
  }

  const denominator = 1 / (va + vb + vc);
  const v = vb * denominator;
  const w = vc * denominator;
  return squaredLength(
    apX - abX * v - acX * w,
    apY - abY * v - acY * w,
    apZ - abZ * v - acZ * w,
  );
}

export function prepareTriangles(mesh: IndexedMesh): PreparedTriangle[] {
  validateIndexedMesh(mesh);
  const triangleCount = mesh.indices.length / 3;
  const triangles: PreparedTriangle[] = [];

  for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
    const indexOffset = triangleIndex * 3;
    const aIndex = mesh.indices[indexOffset]! * 3;
    const bIndex = mesh.indices[indexOffset + 1]! * 3;
    const cIndex = mesh.indices[indexOffset + 2]! * 3;
    triangles.push({
      a: [
        mesh.positions[aIndex]!,
        mesh.positions[aIndex + 1]!,
        mesh.positions[aIndex + 2]!,
      ],
      b: [
        mesh.positions[bIndex]!,
        mesh.positions[bIndex + 1]!,
        mesh.positions[bIndex + 2]!,
      ],
      c: [
        mesh.positions[cIndex]!,
        mesh.positions[cIndex + 1]!,
        mesh.positions[cIndex + 2]!,
      ],
    });
  }

  return triangles;
}
