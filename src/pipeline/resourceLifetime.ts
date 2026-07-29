export async function withAsyncResource<Resource, Result>(
  acquire: () => Promise<Resource>,
  release: (resource: Resource) => void,
  use: (resource: Resource) => Promise<Result>,
): Promise<Result> {
  const resource = await acquire();
  try {
    return await use(resource);
  } finally {
    release(resource);
  }
}
