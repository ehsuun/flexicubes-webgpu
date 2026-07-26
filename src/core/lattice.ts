import type {
  Bounds3,
  GridSize3,
  Lattice3D,
  Vec3,
} from "./types.js";

function assertFiniteVec3(name: string, value: Vec3): void {
  for (let axis = 0; axis < 3; axis++) {
    if (!Number.isFinite(value[axis])) {
      throw new RangeError(`${name}[${axis}] must be finite`);
    }
  }
}

function assertCellCounts(cellCounts: GridSize3): void {
  for (let axis = 0; axis < 3; axis++) {
    const count = cellCounts[axis]!;
    if (!Number.isSafeInteger(count) || count < 1) {
      throw new RangeError(
        `cellCounts[${axis}] must be a positive safe integer`,
      );
    }
  }
}

function checkedProduct(values: GridSize3): number {
  const product = values[0] * values[1] * values[2];
  if (!Number.isSafeInteger(product)) {
    throw new RangeError("lattice sample count exceeds safe integer range");
  }
  return product;
}

export function createLattice3D(
  domain: Bounds3,
  cellCounts: GridSize3,
): Lattice3D {
  assertFiniteVec3("domain.min", domain.min);
  assertFiniteVec3("domain.max", domain.max);
  assertCellCounts(cellCounts);

  const sampleCounts: GridSize3 = [
    cellCounts[0] + 1,
    cellCounts[1] + 1,
    cellCounts[2] + 1,
  ];
  const sampleSpacing: Vec3 = [
    (domain.max[0] - domain.min[0]) / cellCounts[0],
    (domain.max[1] - domain.min[1]) / cellCounts[1],
    (domain.max[2] - domain.min[2]) / cellCounts[2],
  ];

  for (let axis = 0; axis < 3; axis++) {
    if (!(sampleSpacing[axis]! > 0)) {
      throw new RangeError(
        `domain.max[${axis}] must be greater than domain.min[${axis}]`,
      );
    }
  }

  return {
    cellCounts: [...cellCounts],
    sampleCounts,
    sampleOrigin: [...domain.min],
    sampleSpacing,
    sampleCount: checkedProduct(sampleCounts),
  };
}

export function latticeSampleIndex(
  lattice: Lattice3D,
  x: number,
  y: number,
  z: number,
): number {
  const coordinates = [x, y, z];
  for (let axis = 0; axis < 3; axis++) {
    const coordinate = coordinates[axis]!;
    if (
      !Number.isSafeInteger(coordinate)
      || coordinate < 0
      || coordinate >= lattice.sampleCounts[axis]!
    ) {
      throw new RangeError(`sample coordinate ${axis} is outside the lattice`);
    }
  }

  return (
    z * lattice.sampleCounts[0] * lattice.sampleCounts[1]
    + y * lattice.sampleCounts[0]
    + x
  );
}

export function latticeSamplePosition(
  lattice: Lattice3D,
  x: number,
  y: number,
  z: number,
): Vec3 {
  latticeSampleIndex(lattice, x, y, z);
  return [
    lattice.sampleOrigin[0] + x * lattice.sampleSpacing[0],
    lattice.sampleOrigin[1] + y * lattice.sampleSpacing[1],
    lattice.sampleOrigin[2] + z * lattice.sampleSpacing[2],
  ];
}

export function estimateDenseFieldMemoryBytes(lattice: Lattice3D): number {
  return lattice.sampleCount * Float32Array.BYTES_PER_ELEMENT;
}
