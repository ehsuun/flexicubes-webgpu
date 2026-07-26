import { existsSync } from "node:fs";
import { chromium } from "playwright-core";
import { createServer } from "vite";

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
].filter((candidate) => candidate !== undefined);

const executablePath = CHROME_CANDIDATES.find((candidate) => (
  existsSync(candidate)
));
if (executablePath === undefined) {
  throw new Error(
    "Chrome was not found. Set CHROME_PATH to run WebGPU browser tests.",
  );
}

const server = await createServer({
  server: {
    host: "127.0.0.1",
    port: 0,
    strictPort: false,
  },
  logLevel: "error",
});
await server.listen();

const localUrl = server.resolvedUrls?.local[0];
if (localUrl === undefined) {
  await server.close();
  throw new Error("Vite did not expose a local test URL");
}

const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--disable-gpu-sandbox",
  ],
});

try {
  const page = await browser.newPage();
  const browserErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    browserErrors.push(error.message);
  });

  const stressQuery = process.argv.includes("--stress") ? "?stress=1" : "";
  await page.goto(
    `${localUrl}tests/browser/webgpuHarness.html${stressQuery}`,
    { waitUntil: "networkidle" },
  );
  await page.waitForFunction(
    () => document.body.dataset.status !== undefined,
    undefined,
    { timeout: process.argv.includes("--stress") ? 300_000 : 60_000 },
  );
  const status = await page.locator("body").getAttribute("data-status");
  const result = await page.locator("#result").textContent();
  if (status !== "passed") {
    throw new Error(
      `WebGPU browser verification failed: ${result}\n`
      + browserErrors.join("\n"),
    );
  }
  process.stdout.write(`${result}\n`);
} finally {
  await browser.close();
  await server.close();
}
