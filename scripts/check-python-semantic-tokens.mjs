import fs from "node:fs";
import path from "node:path";
import {
  findPythonFences,
  sourceDigest,
} from "./lib/python-semantic-cache.mjs";

const root = process.cwd();
const docsRoot = path.join(root, "docs");
const cachePath = path.join(
  docsRoot,
  ".vitepress",
  "python-semantic-tokens.json",
);

if (!fs.existsSync(cachePath)) {
  throw new Error(
    "缺少 Python 语义高亮缓存；请在 VS Code/Pylance 可用时运行 npm run semantic:refresh",
  );
}

const cache = JSON.parse(fs.readFileSync(cachePath, "utf8"));
const fences = findPythonFences(docsRoot);
const missing = fences.filter(({ key }) => !cache.blocks?.[key]);
const digestMatches = cache.sourceDigest === sourceDigest(fences);
const semanticTypeCounts = {};
for (const block of Object.values(cache.blocks ?? {})) {
  for (const token of block.tokens ?? []) {
    semanticTypeCounts[token.type] = (semanticTypeCounts[token.type] ?? 0) + 1;
  }
}
const requiredTypes = ["class", "function", "method", "variable", "parameter"];
const missingTypes = requiredTypes.filter((type) => !semanticTypeCounts[type]);

if (missing.length || !digestMatches || missingTypes.length) {
  const examples = missing
    .slice(0, 5)
    .map(({ file, line }) => `${file}:${line}`)
    .join(", ");
  throw new Error(
    `Python 语义高亮缓存已过期（缺少 ${missing.length} 个代码块${examples ? `：${examples}` : ""}）；` +
      `请运行 npm run semantic:refresh${missingTypes.length ? `；缺少语义类型：${missingTypes.join(", ")}` : ""}`,
  );
}

console.log(
  `Python 语义高亮缓存有效：${fences.length} 个代码块；` +
    requiredTypes
      .map((type) => `${type}=${semanticTypeCounts[type]}`)
      .join("，") +
    "。",
);
