const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const vscode = require("vscode");

const LEGEND_COMMAND = "vscode.provideDocumentSemanticTokensLegend";
const TOKENS_COMMAND = "vscode.provideDocumentSemanticTokens";

async function run() {
  const root = process.env.HWM_WORKSPACE_ROOT;
  if (!root) throw new Error("HWM_WORKSPACE_ROOT is not set");

  const { CACHE_VERSION, findPythonFences, sourceDigest } = await import(
    path.join(root, "scripts/lib/python-semantic-cache.mjs")
  );
  const docsRoot = path.join(root, "docs");
  const allFences = findPythonFences(docsRoot);
  const unique = [
    ...new Map(allFences.map((fence) => [fence.key, fence])).values(),
  ];
  const requestedLimit = Number.parseInt(
    process.env.HWM_SEMANTIC_LIMIT || "",
    10,
  );
  const fences = Number.isFinite(requestedLimit)
    ? unique.slice(0, requestedLimit)
    : unique;

  const pylance = vscode.extensions.getExtension("ms-python.vscode-pylance");
  if (!pylance) {
    throw new Error(
      "未安装 ms-python.vscode-pylance，无法导出 Python 语义 token",
    );
  }

  await vscode.extensions.getExtension("ms-python.python")?.activate();
  await pylance.activate();

  const tempRoot = await fs.mkdtemp(path.join(root, ".pylance-highlight-"));
  const blocks = {};
  try {
    for (let index = 0; index < fences.length; index += 1) {
      const fence = fences[index];
      const filename = path.join(
        tempRoot,
        `snippet-${index.toString().padStart(4, "0")}.py`,
      );
      await fs.writeFile(filename, `${fence.code}\n`, "utf8");
      const uri = vscode.Uri.file(filename);
      const document = await vscode.workspace.openTextDocument(uri);
      await vscode.window.showTextDocument(document, {
        preview: true,
        preserveFocus: false,
      });

      const { legend, semanticTokens } = await requestSemanticTokens(uri);
      blocks[fence.key] = {
        tokens: decodeTokens(semanticTokens.data, legend),
      };

      if ((index + 1) % 10 === 0 || index + 1 === fences.length) {
        console.log(`[semantic] ${index + 1}/${fences.length}`);
      }
    }
  } finally {
    await vscode.commands.executeCommand("workbench.action.closeAllEditors");
    await fs.rm(tempRoot, { recursive: true, force: true });
  }

  const cachePath =
    process.env.HWM_SEMANTIC_OUTPUT ||
    path.join(docsRoot, ".vitepress", "python-semantic-tokens.json");
  const cache = {
    version: CACHE_VERSION,
    sourceDigest: sourceDigest(allFences),
    generatedBy: {
      vscode: vscode.version,
      pylance: pylance.packageJSON.version,
      platform: `${os.platform()}-${os.arch()}`,
    },
    blocks,
  };
  await fs.writeFile(cachePath, `${JSON.stringify(cache)}\n`, "utf8");
  console.log(
    `[semantic] wrote ${Object.keys(blocks).length} blocks to ${cachePath}`,
  );
}

async function requestSemanticTokens(uri) {
  let lastError;
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const legend = await vscode.commands.executeCommand(LEGEND_COMMAND, uri);
      const semanticTokens = await vscode.commands.executeCommand(
        TOKENS_COMMAND,
        uri,
      );
      if (legend?.tokenTypes && semanticTokens?.data)
        return { legend, semanticTokens };
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(
    `Pylance 未能为 ${uri.fsPath} 提供 semantic tokens${lastError ? `: ${lastError}` : ""}`,
  );
}

function decodeTokens(data, legend) {
  const decoded = [];
  let line = 0;
  let start = 0;
  for (let index = 0; index < data.length; index += 5) {
    const deltaLine = data[index];
    const deltaStart = data[index + 1];
    line += deltaLine;
    start = deltaLine === 0 ? start + deltaStart : deltaStart;
    const modifierBits = data[index + 4];
    const modifiers = legend.tokenModifiers.filter(
      (_modifier, modifierIndex) => (modifierBits & (2 ** modifierIndex)) !== 0,
    );
    decoded.push({
      line,
      start,
      length: data[index + 2],
      type: legend.tokenTypes[data[index + 3]],
      ...(modifiers.length ? { modifiers } : {}),
    });
  }
  return decoded;
}

module.exports = { run };
