import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import MarkdownIt from "markdown-it";

export const CACHE_VERSION = 1;

export function codeKey(code) {
  return crypto.createHash("sha256").update(code.trimEnd()).digest("hex");
}

export function findPythonFences(docsRoot) {
  const md = new MarkdownIt();
  const fences = [];

  for (const file of walkMarkdownFiles(docsRoot)) {
    const source = fs.readFileSync(file, "utf8");
    for (const token of md.parse(source, {})) {
      if (token.type !== "fence") continue;
      const language = token.info.trim().split(/\s+/, 1)[0].toLowerCase();
      if (language !== "python" && language !== "py") continue;

      const code = token.content.trimEnd();
      fences.push({
        code,
        key: codeKey(code),
        file: path.relative(docsRoot, file),
        line: (token.map?.[0] ?? 0) + 2,
      });
    }
  }

  return fences;
}

export function sourceDigest(fences) {
  const entries = fences
    .map(({ file, key }) => `${file}:${key}`)
    .sort()
    .join("\n");
  return crypto.createHash("sha256").update(entries).digest("hex");
}

function walkMarkdownFiles(root) {
  const files = [];
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (entry.name === ".vitepress") continue;
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(file);
      else if (entry.isFile() && entry.name.endsWith(".md")) files.push(file);
    }
  };
  visit(root);
  return files.sort();
}
