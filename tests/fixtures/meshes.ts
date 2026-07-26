import type { IndexedMesh, Vec3 } from "../../src/index.js";

const CUBE_INDICES = [
  0, 2, 1, 1, 2, 3,
  4, 5, 6, 5, 7, 6,
  0, 4, 2, 4, 6, 2,
  1, 3, 5, 5, 3, 7,
  0, 1, 4, 1, 5, 4,
  2, 6, 3, 3, 6, 7,
];

export function createCubeMesh(
  center: Vec3 = [0, 0, 0],
  halfExtent = 0.5,
): IndexedMesh {
  const [centerX, centerY, centerZ] = center;
  const lowX = centerX - halfExtent;
  const lowY = centerY - halfExtent;
  const lowZ = centerZ - halfExtent;
  const highX = centerX + halfExtent;
  const highY = centerY + halfExtent;
  const highZ = centerZ + halfExtent;
  return {
    positions: new Float32Array([
      lowX, lowY, lowZ,
      highX, lowY, lowZ,
      lowX, highY, lowZ,
      highX, highY, lowZ,
      lowX, lowY, highZ,
      highX, lowY, highZ,
      lowX, highY, highZ,
      highX, highY, highZ,
    ]),
    indices: new Uint16Array(CUBE_INDICES),
  };
}

export function createPlaneMesh(): IndexedMesh {
  return {
    positions: new Float32Array([
      -0.75, -0.75, 0,
      0.75, -0.75, 0,
      -0.75, 0.75, 0,
      0.75, 0.75, 0,
    ]),
    indices: new Uint16Array([0, 1, 2, 1, 3, 2]),
  };
}

export function reverseWinding(mesh: IndexedMesh): IndexedMesh {
  const indices = new Uint32Array(mesh.indices.length);
  for (let index = 0; index < mesh.indices.length; index += 3) {
    indices[index] = mesh.indices[index]!;
    indices[index + 1] = mesh.indices[index + 2]!;
    indices[index + 2] = mesh.indices[index + 1]!;
  }
  return {
    positions: mesh.positions.slice(),
    indices,
  };
}

export function mergeMeshes(meshes: readonly IndexedMesh[]): IndexedMesh {
  const positionLength = meshes.reduce(
    (total, mesh) => total + mesh.positions.length,
    0,
  );
  const indexLength = meshes.reduce(
    (total, mesh) => total + mesh.indices.length,
    0,
  );
  const positions = new Float32Array(positionLength);
  const indices = new Uint32Array(indexLength);
  let positionOffset = 0;
  let indexOffset = 0;
  let vertexOffset = 0;

  for (const mesh of meshes) {
    positions.set(mesh.positions, positionOffset);
    for (let index = 0; index < mesh.indices.length; index++) {
      indices[indexOffset + index] = mesh.indices[index]! + vertexOffset;
    }
    positionOffset += mesh.positions.length;
    indexOffset += mesh.indices.length;
    vertexOffset += mesh.positions.length / 3;
  }

  return { positions, indices };
}
