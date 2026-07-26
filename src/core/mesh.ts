import type {
  Bounds3,
  IndexedMesh,
  Matrix4,
  Vec3,
} from "./types.js";

function assertMatrix4(matrix: Matrix4): void {
  if (matrix.length !== 16) {
    throw new RangeError("matrix must contain 16 column-major values");
  }
  for (let index = 0; index < matrix.length; index++) {
    if (!Number.isFinite(matrix[index])) {
      throw new RangeError(`matrix[${index}] must be finite`);
    }
  }
}

export function validateIndexedMesh(mesh: IndexedMesh): void {
  if (mesh.positions.length === 0 || mesh.positions.length % 3 !== 0) {
    throw new RangeError(
      "positions must contain one or more tightly packed XYZ vertices",
    );
  }
  if (mesh.indices.length === 0 || mesh.indices.length % 3 !== 0) {
    throw new RangeError(
      "indices must contain one or more complete triangles",
    );
  }

  for (let index = 0; index < mesh.positions.length; index++) {
    if (!Number.isFinite(mesh.positions[index])) {
      throw new RangeError(`positions[${index}] must be finite`);
    }
  }

  const vertexCount = mesh.positions.length / 3;
  for (let index = 0; index < mesh.indices.length; index++) {
    if (mesh.indices[index]! >= vertexCount) {
      throw new RangeError(
        `indices[${index}] references a missing vertex`,
      );
    }
  }
}

export function computeMeshBounds(mesh: IndexedMesh): Bounds3 {
  validateIndexedMesh(mesh);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (let index = 0; index < mesh.positions.length; index += 3) {
    const x = mesh.positions[index]!;
    const y = mesh.positions[index + 1]!;
    const z = mesh.positions[index + 2]!;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }

  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
  };
}

export function fitSdfDomain(
  mesh: IndexedMesh,
  paddingRatio = 0.12,
  minimumPadding = 0.05,
): Bounds3 {
  if (!Number.isFinite(paddingRatio) || paddingRatio < 0) {
    throw new RangeError("paddingRatio must be finite and non-negative");
  }
  if (!Number.isFinite(minimumPadding) || minimumPadding <= 0) {
    throw new RangeError("minimumPadding must be finite and positive");
  }

  const bounds = computeMeshBounds(mesh);
  const size: Vec3 = [
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ];
  const maximumExtent = Math.max(...size);
  const padding = Math.max(maximumExtent * paddingRatio, minimumPadding);

  return {
    min: [
      bounds.min[0] - padding,
      bounds.min[1] - padding,
      bounds.min[2] - padding,
    ],
    max: [
      bounds.max[0] + padding,
      bounds.max[1] + padding,
      bounds.max[2] + padding,
    ],
  };
}

export function transformIndexedMesh(
  mesh: IndexedMesh,
  matrix: Matrix4,
): IndexedMesh {
  validateIndexedMesh(mesh);
  assertMatrix4(matrix);

  const positions = new Float32Array(mesh.positions.length);
  for (let index = 0; index < mesh.positions.length; index += 3) {
    const x = mesh.positions[index]!;
    const y = mesh.positions[index + 1]!;
    const z = mesh.positions[index + 2]!;
    const w = (
      matrix[3]! * x
      + matrix[7]! * y
      + matrix[11]! * z
      + matrix[15]!
    );
    if (Math.abs(w) < Number.EPSILON) {
      throw new RangeError("matrix transformed a position to w = 0");
    }

    positions[index] = (
      matrix[0]! * x
      + matrix[4]! * y
      + matrix[8]! * z
      + matrix[12]!
    ) / w;
    positions[index + 1] = (
      matrix[1]! * x
      + matrix[5]! * y
      + matrix[9]! * z
      + matrix[13]!
    ) / w;
    positions[index + 2] = (
      matrix[2]! * x
      + matrix[6]! * y
      + matrix[10]! * z
      + matrix[14]!
    ) / w;
  }

  return {
    positions,
    indices: mesh.indices,
  };
}
