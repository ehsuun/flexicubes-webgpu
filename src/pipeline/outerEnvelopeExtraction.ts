import type {
  IndexedMesh,
  ProxyOuterEnvelopeEvidence,
  ProxyOuterEnvelopeOptions,
  ProxyOuterEnvelopeVerificationOptions,
  WebGpuFlexiCubesExtractionResult,
} from "../core/types.js";
import {
  ProxyOuterEnvelopeError,
  verifyProxyOuterEnvelopeWithProgress,
} from "./outerEnvelope.js";

const DEFAULT_MAXIMUM_ATTEMPTS = 3;
const EXPANSION_SAFETY_FACTOR = 1.25;

export type OuterEnvelopeExtractor = (
  isoValue: number,
) => Promise<WebGpuFlexiCubesExtractionResult>;

export type OuterEnvelopeExtractionResult = Readonly<{
  extraction: WebGpuFlexiCubesExtractionResult;
  evidence: ProxyOuterEnvelopeEvidence;
}>;

const maximumAttempts = (value: number | undefined): number => {
  const resolved = value ?? DEFAULT_MAXIMUM_ATTEMPTS;
  if (!Number.isSafeInteger(resolved) || resolved < 1) {
    throw new RangeError("maximumAttempts must be a positive safe integer");
  }
  return resolved;
};

const validateEnvelope = (options: ProxyOuterEnvelopeOptions): void => {
  if (!Number.isFinite(options.minimumSeparation)
    || options.minimumSeparation < 0) {
    throw new RangeError(
      "minimumSeparation must be finite and non-negative",
    );
  }
  if (!Number.isFinite(options.maximumExpansion)
    || options.maximumExpansion < 0) {
    throw new RangeError("maximumExpansion must be finite and non-negative");
  }
};

const verificationOptions = (
  options: ProxyOuterEnvelopeOptions,
  signal: AbortSignal | undefined,
): ProxyOuterEnvelopeVerificationOptions => ({
  minimumSeparation: options.minimumSeparation,
  ...(options.maximumSourceSamples === undefined
    ? {}
    : { maximumSourceSamples: options.maximumSourceSamples }),
  ...(options.sampleBatchSize === undefined
    ? {}
    : { sampleBatchSize: options.sampleBatchSize }),
  ...(signal === undefined ? {} : { signal }),
});

const evidence = (
  startedAt: number,
  attempts: number,
  initialIsoValue: number,
  finalIsoValue: number,
  initialVerification: ProxyOuterEnvelopeEvidence["initialVerification"],
  finalVerification: ProxyOuterEnvelopeEvidence["finalVerification"],
): ProxyOuterEnvelopeEvidence => ({
  attempts,
  initialIsoValue,
  finalIsoValue,
  initialVerification,
  finalVerification,
  elapsedMs: performance.now() - startedAt,
});

export const extractProxyOuterEnvelope = async (
  source: IndexedMesh,
  initialIsoValue: number,
  options: ProxyOuterEnvelopeOptions,
  signal: AbortSignal | undefined,
  extract: OuterEnvelopeExtractor,
): Promise<OuterEnvelopeExtractionResult> => {
  validateEnvelope(options);
  if (!Number.isFinite(initialIsoValue)) {
    throw new RangeError("initialIsoValue must be finite");
  }
  const attemptLimit = maximumAttempts(options.maximumAttempts);
  const maximumIsoValue = initialIsoValue + options.maximumExpansion;
  const startedAt = performance.now();
  let isoValue = initialIsoValue;
  let initialVerification:
    | ProxyOuterEnvelopeEvidence["initialVerification"]
    | undefined;
  for (let attempt = 1; attempt <= attemptLimit; attempt++) {
    const extraction = await extract(isoValue);
    if (extraction.stats.boundarySurfaceEdgeCount > 0) {
      throw new RangeError(
        "outer-envelope extraction requires a closed proxy surface",
      );
    }
    const verification = await verifyProxyOuterEnvelopeWithProgress(
      source,
      extraction.mesh,
      verificationOptions(options, signal),
      (completedSamples, totalSamples) => options.onProgress?.({
        attempt,
        maximumAttempts: attemptLimit,
        isoValue,
        completedSamples,
        totalSamples,
      }),
    );
    initialVerification ??= verification;
    const currentEvidence = evidence(
      startedAt,
      attempt,
      initialIsoValue,
      isoValue,
      initialVerification,
      verification,
    );
    if (verification.violationCount === 0) {
      return { extraction, evidence: currentEvidence };
    }
    const requiredStep = verification.maximumIngress
      * EXPANSION_SAFETY_FACTOR;
    const nextIsoValue = Math.min(
      maximumIsoValue,
      isoValue + requiredStep,
    );
    if (
      attempt === attemptLimit
      || nextIsoValue <= isoValue
    ) {
      throw new ProxyOuterEnvelopeError(
        [
          "proxy did not satisfy sampled outer-envelope containment",
          `after ${attempt} attempt(s)`,
          `at iso ${isoValue}`,
          `with ${verification.violationCount}/`
            + `${verification.sourceSampleCount} violating samples`,
          `and maximum ingress ${verification.maximumIngress}`,
        ].join(" "),
        currentEvidence,
      );
    }
    isoValue = nextIsoValue;
  }
  throw new Error("outer-envelope attempt loop exited unexpectedly");
};
