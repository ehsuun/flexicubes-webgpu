import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const [, , sourceArgument, commitArgument] = process.argv;
if (sourceArgument === undefined || commitArgument === undefined) {
  throw new Error(
    "Usage: node scripts/generate-flexicubes-tables.mjs "
    + "<upstream tables.py> <upstream commit>",
  );
}

const sourcePath = resolve(sourceArgument);
const source = await readFile(sourcePath, "utf8");

function readPythonList(name) {
  const assignment = source.indexOf(`${name} =`);
  if (assignment < 0) {
    throw new Error(`Could not find ${name} in ${sourcePath}`);
  }
  const start = source.indexOf("[", assignment);
  let depth = 0;
  for (let index = start; index < source.length; index++) {
    const character = source[index];
    if (character === "[") {
      depth++;
    } else if (character === "]") {
      depth--;
      if (depth === 0) {
        return JSON.parse(source.slice(start, index + 1));
      }
    }
  }
  throw new Error(`Could not find the end of ${name}`);
}

function flatten(values) {
  return values.flat(Infinity);
}

function formatValues(values, perLine) {
  const lines = [];
  for (let index = 0; index < values.length; index += perLine) {
    lines.push(`  ${values.slice(index, index + perLine).join(", ")},`);
  }
  return lines.join("\n");
}

const dmcTable = readPythonList("dmc_table");
const dualVertexCounts = readPythonList("num_vd_table");
const ambiguityChecks = readPythonList("check_table");
if (
  dmcTable.length !== 256
  || dualVertexCounts.length !== 256
  || ambiguityChecks.length !== 256
) {
  throw new Error("Expected 256 FlexiCubes cases");
}

const output = `/*
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified for flexicubes-webgpu:
 * - generated from nv-tlabs/FlexiCubes tables.py at ${commitArgument}
 * - flattened into typed arrays for renderer-neutral TypeScript lookup
 * - tetrahedral-output tables were intentionally omitted
 */

export const FLEXICUBES_CASE_COUNT = 256;
export const FLEXICUBES_MAX_DUAL_VERTICES = 4;
export const FLEXICUBES_EDGES_PER_DUAL_VERTEX = 7;

export const FLEXICUBES_DMC_TABLE = new Int8Array([
${formatValues(flatten(dmcTable), 28)}
]);

export const FLEXICUBES_DUAL_VERTEX_COUNTS = new Uint8Array([
${formatValues(flatten(dualVertexCounts), 32)}
]);

export const FLEXICUBES_AMBIGUITY_CHECKS = new Int16Array([
${formatValues(flatten(ambiguityChecks), 25)}
]);
`;

await writeFile(
  new URL("../src/flexicubes/tables.ts", import.meta.url),
  output,
  "utf8",
);
