import type { ExtractedMesh } from "../core/types.js";

type Triangle = readonly [number, number, number];

function compareNumber(left: number, right: number): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareVertex(
  mesh: ExtractedMesh,
  left: number,
  right: number,
): number {
  const sourceCellOrder = compareNumber(
    mesh.sourceCells[left]!,
    mesh.sourceCells[right]!,
  );
  if (sourceCellOrder !== 0) {
    return sourceCellOrder;
  }
  for (let axis = 0; axis < 3; axis++) {
    const coordinateOrder = compareNumber(
      mesh.positions[left * 3 + axis]!,
      mesh.positions[right * 3 + axis]!,
    );
    if (coordinateOrder !== 0) {
      return coordinateOrder;
    }
  }
  return left - right;
}

function rotateTriangleToMinimum(
  first: number,
  second: number,
  third: number,
): Triangle {
  if (first <= second && first <= third) {
    return [first, second, third];
  }
  if (second <= third) {
    return [second, third, first];
  }
  return [third, first, second];
}

function compareTriangle(left: Triangle, right: Triangle): number {
  for (let corner = 0; corner < 3; corner++) {
    const order = compareNumber(left[corner]!, right[corner]!);
    if (order !== 0) {
      return order;
    }
  }
  return 0;
}

export function canonicalizeExtractedMesh(
  mesh: ExtractedMesh,
): ExtractedMesh {
  const vertexCount = mesh.positions.length / 3;
  const orderedVertices = Array.from(
    { length: vertexCount },
    (_, index) => index,
  ).sort((left, right) => compareVertex(mesh, left, right));
  const remap = new Uint32Array(vertexCount);
  const positions = new Float32Array(mesh.positions.length);
  const sourceCells = new Uint32Array(mesh.sourceCells.length);
  for (let target = 0; target < orderedVertices.length; target++) {
    const source = orderedVertices[target]!;
    remap[source] = target;
    sourceCells[target] = mesh.sourceCells[source]!;
    for (let axis = 0; axis < 3; axis++) {
      positions[target * 3 + axis] = mesh.positions[source * 3 + axis]!;
    }
  }

  const triangles: Triangle[] = [];
  for (let offset = 0; offset < mesh.indices.length; offset += 3) {
    triangles.push(rotateTriangleToMinimum(
      remap[mesh.indices[offset]!]!,
      remap[mesh.indices[offset + 1]!]!,
      remap[mesh.indices[offset + 2]!]!,
    ));
  }
  triangles.sort(compareTriangle);
  const indices = new Uint32Array(mesh.indices.length);
  for (let triangle = 0; triangle < triangles.length; triangle++) {
    const value = triangles[triangle]!;
    indices[triangle * 3] = value[0];
    indices[triangle * 3 + 1] = value[1];
    indices[triangle * 3 + 2] = value[2];
  }
  return { positions, indices, sourceCells };
}
