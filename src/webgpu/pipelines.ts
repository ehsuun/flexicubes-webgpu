const PIPELINES = new WeakMap<
GPUDevice,
Map<string, Promise<GPUComputePipeline>>
>();

async function compileComputePipeline(
  device: GPUDevice,
  label: string,
  code: string,
): Promise<GPUComputePipeline> {
  const module = device.createShaderModule({ label: `${label} shader`, code });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(
    (message) => message.type === "error",
  );
  if (errors.length > 0) {
    const details = errors.map(
      (error) => `${error.lineNum}:${error.linePos} ${error.message}`,
    ).join("\n");
    throw new Error(`${label} WGSL compilation failed:\n${details}`);
  }
  return device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

export function cachedComputePipeline(
  device: GPUDevice,
  key: string,
  label: string,
  code: string,
): Promise<GPUComputePipeline> {
  let devicePipelines = PIPELINES.get(device);
  if (devicePipelines === undefined) {
    devicePipelines = new Map();
    PIPELINES.set(device, devicePipelines);
  }

  const existing = devicePipelines.get(key);
  if (existing !== undefined) {
    return existing;
  }
  const pipeline = compileComputePipeline(device, label, code);
  devicePipelines.set(key, pipeline);
  return pipeline;
}
