import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runTests } from "@vscode/test-electron";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const extensionRoot = path.join(root, "tools", "vscode-semantic-exporter");
const extensionTestsPath = path.join(extensionRoot, "run.cjs");

const candidates = [
  process.env.VSCODE_EXECUTABLE_PATH,
  "/Applications/Visual Studio Code.app/Contents/MacOS/Code",
  "/Applications/Visual Studio Code - Insiders.app/Contents/MacOS/Code - Insiders",
].filter(Boolean);
const vscodeExecutablePath = candidates.find((candidate) =>
  fs.existsSync(candidate),
);

if (!vscodeExecutablePath) {
  throw new Error(
    "找不到 VS Code。请设置 VSCODE_EXECUTABLE_PATH，指向 VS Code 的 Electron 可执行文件。",
  );
}

const userDataDirectory = fs.mkdtempSync(
  path.join(os.tmpdir(), "hwm-vscode-semantic-"),
);
try {
  await runTests({
    vscodeExecutablePath,
    extensionDevelopmentPath: extensionRoot,
    extensionTestsPath,
    reuseMachineInstall: true,
    launchArgs: [
      root,
      `--user-data-dir=${userDataDirectory}`,
      `--extensions-dir=${path.join(process.env.HOME, ".vscode", "extensions")}`,
      "--disable-gpu",
      "--skip-welcome",
      "--skip-release-notes",
    ],
    extensionTestsEnv: {
      HWM_WORKSPACE_ROOT: root,
      HWM_SEMANTIC_LIMIT: process.env.HWM_SEMANTIC_LIMIT,
      HWM_SEMANTIC_OUTPUT: process.env.HWM_SEMANTIC_LIMIT
        ? path.join(userDataDirectory, "python-semantic-tokens.partial.json")
        : undefined,
    },
  });
} finally {
  fs.rmSync(userDataDirectory, { recursive: true, force: true });
}
