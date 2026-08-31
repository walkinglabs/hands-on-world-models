import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { codeKey } from "../../scripts/lib/python-semantic-cache.mjs";

const directory = path.dirname(fileURLToPath(import.meta.url));
const cachePath = path.join(directory, "python-semantic-tokens.json");
const cache = fs.existsSync(cachePath)
  ? JSON.parse(fs.readFileSync(cachePath, "utf8"))
  : { blocks: {} };

const semanticScopes = {
  namespace: "entity.name.namespace",
  module: "entity.name.namespace",
  type: "entity.name.type",
  class: "entity.name.type.class",
  enum: "entity.name.type.enum",
  interface: "entity.name.type.interface",
  struct: "entity.name.type.struct",
  typeParameter: "entity.name.type.parameter",
  parameter: "variable.parameter",
  variable: "variable.other.readwrite",
  property: "variable.other.property",
  enumMember: "variable.other.enummember",
  event: "variable.other.event",
  function: "entity.name.function",
  method: "entity.name.function.member",
  macro: "entity.name.function.preprocessor",
  label: "entity.name.label",
  comment: "comment",
  string: "string",
  keyword: "keyword",
  number: "constant.numeric",
  regexp: "string.regexp",
  operator: "keyword.operator",
  decorator: "entity.name.function.decorator",
  intrinsic: "keyword.operator",
  selfParameter: "variable.parameter.function.language.special.self.python",
  clsParameter: "variable.parameter.function.language.special.cls.python",
  magicFunction: "support.function.magic.python",
  builtinConstant: "constant.language.python",
  parenthesis: "source.python",
  bracket: "source.python",
  curlybrace: "source.python",
  colon: "punctuation.separator.colon.python",
  semicolon: "source.python",
  arrow: "punctuation.separator.annotation.result.python",
};

const scopeNames = [
  ...new Set([
    ...Object.values(semanticScopes),
    "variable.parameter.function-call.python",
    "constant.character.escape.python",
    "comment.typehint.type.notation.python",
    "invalid.illegal",
    "meta.function.decorator.python",
  ]),
];
const stylesByScope = new Map();

export async function setupPythonSemanticStyles(highlighter) {
  await highlighter.loadLanguage({
    name: "hwm-semantic-scopes",
    scopeName: "source.hwm-semantic-scopes",
    patterns: scopeNames.map((scope, index) => ({
      match: `\\bS${index}\\b`,
      name: scope,
    })),
  });

  const source = scopeNames.map((_scope, index) => `S${index}`).join("\n");
  const result = highlighter.codeToTokens(source, {
    lang: "hwm-semantic-scopes",
    themes: { light: "light-plus", dark: "dark-plus" },
    defaultColor: false,
  });

  result.tokens.forEach((line, index) => {
    const token = line.find((candidate) => candidate.content.trim());
    if (token?.htmlStyle) stylesByScope.set(scopeNames[index], token.htmlStyle);
  });
}

export function pythonSemanticTransformer() {
  return {
    name: "hwm:pylance-semantic-tokens",
    tokens(themedTokens) {
      if (this.options.lang !== "python" && this.options.lang !== "py") return;
      const block = cache.blocks?.[codeKey(this.source)];
      if (!block?.tokens?.length) return;

      const byLine = new Map();
      for (const token of block.tokens) {
        const line = byLine.get(token.line) ?? [];
        line.push(token);
        byLine.set(token.line, line);
      }

      return themedTokens.map((lineTokens, lineNumber) => {
        const semanticLine = byLine.get(lineNumber);
        return semanticLine?.length
          ? mergeLine(lineTokens, semanticLine)
          : lineTokens;
      });
    },
  };
}

function mergeLine(lineTokens, semanticTokens) {
  const merged = [];
  let column = 0;

  for (const token of lineTokens) {
    const tokenStart = column;
    const tokenEnd = tokenStart + token.content.length;
    const boundaries = new Set([tokenStart, tokenEnd]);
    for (const semantic of semanticTokens) {
      const semanticEnd = semantic.start + semantic.length;
      if (semantic.start > tokenStart && semantic.start < tokenEnd)
        boundaries.add(semantic.start);
      if (semanticEnd > tokenStart && semanticEnd < tokenEnd)
        boundaries.add(semanticEnd);
    }

    const points = [...boundaries].sort((a, b) => a - b);
    for (let index = 0; index < points.length - 1; index += 1) {
      const start = points[index];
      const end = points[index + 1];
      const semantic = semanticTokens.find(
        (candidate) =>
          candidate.start <= start && candidate.start + candidate.length >= end,
      );
      const style = semantic ? semanticStyle(semantic) : undefined;
      merged.push({
        ...token,
        content: token.content.slice(start - tokenStart, end - tokenStart),
        offset: token.offset + start - tokenStart,
        ...(style
          ? {
              htmlStyle: {
                ...(typeof token.htmlStyle === "object" ? token.htmlStyle : {}),
                ...style,
              },
              htmlAttrs: {
                ...token.htmlAttrs,
                "data-semantic-token": semantic.type,
              },
            }
          : {}),
      });
    }
    column = tokenEnd;
  }

  return merged;
}

function semanticStyle(token) {
  let scope = semanticScopes[token.type];
  const modifiers = new Set(token.modifiers ?? []);

  if (modifiers.has("keywordArgument") && token.type === "parameter") {
    scope = "variable.parameter.function-call.python";
  } else if (modifiers.has("escapeCharacter")) {
    scope = "constant.character.escape.python";
  } else if (modifiers.has("typeHintComment")) {
    scope = "comment.typehint.type.notation.python";
  } else if (modifiers.has("invalid")) {
    scope = "invalid.illegal";
  } else if (
    modifiers.has("decorator") &&
    (token.type === "function" || token.type === "class")
  ) {
    scope = "meta.function.decorator.python";
  }

  return stylesByScope.get(scope);
}
