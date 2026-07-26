import {
  createLattice3D,
  latticeSampleIndex,
  latticeSamplePosition,
  type Bounds3,
  type DenseScalarField3D,
  type GridSize3,
  type Vec3,
} from "../../src/index.js";

export function createAnalyticField(
  domain: Bounds3,
  cellCounts: GridSize3,
  sample: (position: Vec3) => number,
): DenseScalarField3D {
  const lattice = createLattice3D(domain, cellCounts);
  const values = new Float32Array(lattice.sampleCount);
  for (let z = 0; z < lattice.sampleCounts[2]; z++) {
    for (let y = 0; y < lattice.sampleCounts[1]; y++) {
      for (let x = 0; x < lattice.sampleCounts[0]; x++) {
        values[latticeSampleIndex(lattice, x, y, z)] = sample(
          latticeSamplePosition(lattice, x, y, z),
        );
      }
    }
  }
  return {
    ...lattice,
    storage: "cpu-dense",
    layout: "x-fastest",
    values,
    signConvention: "negative-inside",
  };
}

export function createSphereField(
  cells: number,
  radius = 0.62,
): DenseScalarField3D {
  return createAnalyticField(
    { min: [-1, -1, -1], max: [1, 1, 1] },
    [cells, cells, cells],
    ([x, y, z]) => Math.hypot(x, y, z) - radius,
  );
}

export function createSingleCaseField(caseId: number): DenseScalarField3D {
  const field = createAnalyticField(
    { min: [0, 0, 0], max: [1, 1, 1] },
    [1, 1, 1],
    () => 1,
  );
  for (let corner = 0; corner < 8; corner++) {
    field.values[corner] = (caseId & (1 << corner)) === 0 ? 1 : -1;
  }
  return field;
}

export function createAdjacentXCaseField(
  leftCaseId: number,
  rightCaseId: number,
): DenseScalarField3D {
  const field = createAnalyticField(
    { min: [0, 0, 0], max: [2, 1, 1] },
    [2, 1, 1],
    () => 1,
  );
  const assigned = new Uint8Array(field.sampleCount);
  const writeCase = (cellX: number, caseId: number): void => {
    for (let corner = 0; corner < 8; corner++) {
      const x = cellX + (corner & 1);
      const y = (corner >> 1) & 1;
      const z = (corner >> 2) & 1;
      const index = latticeSampleIndex(field, x, y, z);
      const value = (caseId & (1 << corner)) === 0 ? 1 : -1;
      if (assigned[index] === 1 && field.values[index] !== value) {
        throw new Error("adjacent cases disagree on their shared face");
      }
      field.values[index] = value;
      assigned[index] = 1;
    }
  };
  writeCase(0, leftCaseId);
  writeCase(1, rightCaseId);
  return field;
}
