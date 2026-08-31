import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const ROOT = process.cwd();
const DOCS = path.join(ROOT, "docs");
const FIX = process.argv.includes("--fix");
const ONLINE = process.argv.includes("--online");

// Prefer the paper itself (DOI, proceedings, arXiv, or the publisher's page)
// over a secondary summary. Keys follow the author-year spelling used in text.
const SOURCES = new Map(
  Object.entries({
    "Abadi et al., 2016":
      "https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi",
    "Akkaya et al., 2019": "https://arxiv.org/abs/1910.07113",
    "Alemi et al., 2018": "https://arxiv.org/abs/1711.00464",
    "Andrychowicz et al., 2020": "https://doi.org/10.1177/0278364919887447",
    "Argall et al., 2009": "https://doi.org/10.1016/j.robot.2008.10.024",
    "Assran et al., 2023": "https://arxiv.org/abs/2301.08243",
    "Babaeizadeh et al., 2017": "https://arxiv.org/abs/1710.11252",
    "Baevski et al., 2022": "https://arxiv.org/abs/2202.03555",
    "Bardes et al., 2021": "https://arxiv.org/abs/2105.04906",
    "Bardes et al., 2024": "https://arxiv.org/abs/2404.08471",
    "Bellman, 1957":
      "https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming",
    "Bengio et al., 2003": "https://www.jmlr.org/papers/v3/bengio03a.html",
    "Bergstra et al., 2010": "https://doi.org/10.25080/Majora-92bf1922-003",
    "Bishop, 1994":
      "https://www.microsoft.com/en-us/research/publication/mixture-density-networks/",
    "Blattmann et al., 2023": "https://arxiv.org/abs/2311.15127",
    "Botev et al., 2013": "https://doi.org/10.1016/B978-0-444-53859-8.00003-5",
    "Brohan et al., 2022": "https://arxiv.org/abs/2212.06817",
    "Brohan et al., 2023": "https://arxiv.org/abs/2307.15818",
    "Brooks, 1986": "https://doi.org/10.1109/JRA.1986.1087032",
    "Bruce et al., 2024": "https://arxiv.org/abs/2402.15391",
    "Camacho & Bordons, 1999":
      "https://link.springer.com/book/10.1007/978-1-4471-3398-8",
    "Chen et al., 2020": "https://arxiv.org/abs/2002.05709",
    "Chi et al., 2023": "https://arxiv.org/abs/2303.04137",
    "Cho et al., 2014": "https://arxiv.org/abs/1406.1078",
    "Chua et al., 2018": "https://arxiv.org/abs/1805.12114",
    "Chung et al., 2015": "https://arxiv.org/abs/1506.02216",
    "Deng et al., 2009": "https://doi.org/10.1109/CVPR.2009.5206848",
    "Denton & Fergus, 2018": "https://arxiv.org/abs/1802.07687",
    "Devlin et al., 2018": "https://arxiv.org/abs/1810.04805",
    "Dosovitskiy et al., 2020": "https://arxiv.org/abs/2010.11929",
    "Elman, 1990": "https://doi.org/10.1016/0364-0213(90)90002-E",
    "Finn et al., 2016": "https://arxiv.org/abs/1605.07157",
    "Friston, 2010": "https://doi.org/10.1038/nrn2787",
    "Garcia et al., 1989": "https://doi.org/10.1016/0005-1098(89)90002-2",
    "Goodfellow et al., 2014": "https://arxiv.org/abs/1406.2661",
    "Grill et al., 2020": "https://arxiv.org/abs/2006.07733",
    "Gu et al., 2021": "https://arxiv.org/abs/2111.00396",
    "Ha & Schmidhuber, 2018": "https://arxiv.org/abs/1803.10122",
    "Hafner et al., 2019": "https://arxiv.org/abs/1811.04551",
    "Hafner et al., 2020": "https://arxiv.org/abs/1912.01603",
    "Hafner et al., 2021": "https://arxiv.org/abs/2010.02193",
    "Hafner et al., 2023": "https://arxiv.org/abs/2301.04104",
    "Hartley & Zisserman, 2003":
      "https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A",
    "He et al., 2015": "https://arxiv.org/abs/1512.03385",
    "He et al., 2020": "https://arxiv.org/abs/1911.05722",
    "He et al., 2022": "https://arxiv.org/abs/2111.06377",
    "Heusel et al., 2017": "https://arxiv.org/abs/1706.08500",
    "Ho & Salimans, 2022": "https://arxiv.org/abs/2207.12598",
    "Ho et al., 2020": "https://arxiv.org/abs/2006.11239",
    "Hochreiter & Schmidhuber, 1997":
      "https://doi.org/10.1162/neco.1997.9.8.1735",
    "Howell et al., 2022": "https://arxiv.org/abs/2212.00541",
    "Hu et al., 2021": "https://arxiv.org/abs/2106.09685",
    "Hu et al., 2022": "https://arxiv.org/abs/2203.08104",
    "Anthony Hu et al., 2023": "https://arxiv.org/abs/2309.17080",
    "Yihan Hu et al., 2023": "https://arxiv.org/abs/2212.10156",
    "Huang et al., 2023": "https://arxiv.org/abs/2302.07817",
    "Hwangbo et al., 2019": "https://doi.org/10.1126/scirobotics.aau5872",
    "Jia et al., 2014": "https://arxiv.org/abs/1408.5093",
    "Kalman, 1960": "https://doi.org/10.1115/1.3662552",
    "Kerbl et al., 2023": "https://arxiv.org/abs/2308.04079",
    "Khatib, 1987": "https://doi.org/10.1177/027836498700600103",
    "Kim et al., 2024": "https://arxiv.org/abs/2406.09246",
    "Kingma & Welling, 2013": "https://arxiv.org/abs/1312.6114",
    "Kondratyuk et al., 2023": "https://arxiv.org/abs/2312.14125",
    "Krizhevsky et al., 2012":
      "https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
    "Kuindersma et al., 2016": "https://doi.org/10.1007/s10514-015-9479-3",
    "Kumar et al., 2021": "https://arxiv.org/abs/2107.04034",
    "Kwon et al., 2023": "https://arxiv.org/abs/2309.06180",
    "Lambert et al., 2020": "https://arxiv.org/abs/2002.04523",
    "LeCun et al., 1989": "https://doi.org/10.1162/neco.1989.1.4.541",
    "LeCun, 2022": "https://openreview.net/forum?id=BZ5a1r-kVsf",
    "Lee et al., 2020": "https://arxiv.org/abs/2010.11251",
    "Levine et al., 2016": "https://arxiv.org/abs/1504.00702",
    "Li et al., 2022": "https://arxiv.org/abs/2203.17270",
    "Lin, 1992": "https://doi.org/10.1007/BF00992699",
    "Luo et al., 2024": "https://arxiv.org/abs/2401.16013",
    "Makoviychuk et al., 2021": "https://arxiv.org/abs/2108.10470",
    "Markov, 1906": "https://www.mathnet.ru/eng/im8054",
    "McCulloch & Pitts, 1943": "https://doi.org/10.1007/BF02478259",
    "Mildenhall et al., 2020": "https://arxiv.org/abs/2003.08934",
    "Mirza & Osindero, 2014": "https://arxiv.org/abs/1411.1784",
    "Mnih et al., 2013": "https://arxiv.org/abs/1312.5602",
    "Mnih et al., 2015": "https://doi.org/10.1038/nature14236",
    "Nair & Hinton, 2010": "https://icml.cc/Conferences/2010/papers/432.pdf",
    "Nair et al., 2020": "https://arxiv.org/abs/2006.09359",
    "Oh et al., 2015": "https://arxiv.org/abs/1507.08750",
    "OpenAI, 2024":
      "https://openai.com/index/video-generation-models-as-world-simulators/",
    "Open X-Embodiment Collaboration, 2023": "https://arxiv.org/abs/2310.08864",
    "Paszke et al., 2019":
      "https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html",
    "Peebles & Xie, 2023": "https://arxiv.org/abs/2212.09748",
    "Peng et al., 2018": "https://arxiv.org/abs/1808.00177",
    "Philion & Fidler, 2020": "https://arxiv.org/abs/2008.05711",
    "Pinto et al., 2017": "https://arxiv.org/abs/1710.06542",
    "Pomerleau, 1989":
      "https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html",
    "Pope et al., 2022": "https://arxiv.org/abs/2211.05102",
    "Radford et al., 2018":
      "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
    "Rajeswaran et al., 2017": "https://arxiv.org/abs/1709.10087",
    "Richalet et al., 1978": "https://doi.org/10.1016/0005-1098(78)90001-8",
    "Robbins & Monro, 1951": "https://doi.org/10.1214/aoms/1177729586",
    "Rombach et al., 2022": "https://arxiv.org/abs/2112.10752",
    "Rosenblatt, 1958": "https://doi.org/10.1037/h0042519",
    "Ross & Bagnell, 2010": "https://proceedings.mlr.press/v9/ross10a.html",
    "Ross et al., 2011": "https://proceedings.mlr.press/v15/ross11a.html",
    "Salisbury & Craig, 1982": "https://doi.org/10.1177/027836498200100102",
    "Rubinstein, 1997": "https://doi.org/10.1016/S0377-2217(96)00385-2",
    "Rumelhart et al., 1986": "https://doi.org/10.1038/323533a0",
    "Salimans et al., 2016": "https://arxiv.org/abs/1606.03498",
    "Schrittwieser et al., 2020": "https://arxiv.org/abs/1911.08265",
    "Schulman et al., 2017": "https://arxiv.org/abs/1707.06347",
    "Sentis et al., 2007": "https://doi.org/10.1109/ROBOT.2007.363998",
    "Shannon, 1948": "https://doi.org/10.1002/j.1538-7305.1948.tb01338.x",
    "Silver et al., 2016": "https://doi.org/10.1038/nature16961",
    "Silver et al., 2017": "https://doi.org/10.1038/nature24270",
    "Silver et al., 2018": "https://doi.org/10.1126/science.aar6404",
    "Sohl-Dickstein et al., 2015": "https://arxiv.org/abs/1503.03585",
    "Sutton & Barto, 1998":
      "http://incompleteideas.net/book/first/the-book.html",
    "Sutton et al., 1999":
      "https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html",
    "Sutton, 1988": "https://doi.org/10.1007/BF00115009",
    "Sutton, 1990": "https://dl.acm.org/doi/10.5555/645530.658292",
    "Talvitie, 2014": "https://ojs.aaai.org/index.php/AAAI/article/view/8852",
    "Tobin et al., 2017": "https://arxiv.org/abs/1703.06907",
    "Todorov et al., 2012": "https://doi.org/10.1109/IROS.2012.6386109",
    "Tong et al., 2022": "https://arxiv.org/abs/2203.12602",
    "Unterthiner et al., 2018": "https://arxiv.org/abs/1812.01717",
    "van den Oord et al., 2017": "https://arxiv.org/abs/1711.00937",
    "Vaswani et al., 2017": "https://arxiv.org/abs/1706.03762",
    "Wei et al., 2023": "https://arxiv.org/abs/2303.09551",
    "Williams, 1992": "https://doi.org/10.1007/BF00992696",
    "Wang et al., 2026": "https://arxiv.org/abs/2605.12090",
    "Xiaofeng Wang et al., 2023a": "https://arxiv.org/abs/2309.09777",
    "Xiaofeng Wang et al., 2023b": "https://arxiv.org/abs/2303.03991",
    "Shi et al., 2015": "https://arxiv.org/abs/1506.04214",
    "Yu et al., 2023": "https://arxiv.org/abs/2310.05737",
    "Zhang & Agrawala, 2023": "https://arxiv.org/abs/2302.05543",
    "Zhao et al., 2023": "https://arxiv.org/abs/2304.13705",
    "Zheng et al., 2023": "https://arxiv.org/abs/2311.16038",
    "Åström, 1965": "https://doi.org/10.1016/0022-247X(65)90154-X",
  }),
);

const NORMALIZE = new Map([
  ["Ha and Schmidhuber, 2018", "Ha & Schmidhuber, 2018"],
  ["Ha et al., 2018", "Ha & Schmidhuber, 2018"],
  ["Hafner et al., 2018", "Hafner et al., 2019"],
  ["McCulloch et al., 1943", "McCulloch & Pitts, 1943"],
  ["He et al., 2021", "He et al., 2022"],
  ["Hartley et al., 2003", "Hartley & Zisserman, 2003"],
  ["Padalkar et al., 2023", "Open X-Embodiment Collaboration, 2023"],
  ["Ross and Bagnell, 2010", "Ross & Bagnell, 2010"],
  ["Van Den Oord et al., 2017", "van den Oord et al., 2017"],
  ["Xingjian et al., 2015", "Shi et al., 2015"],
]);

function sourceFor(label, context, file) {
  return SOURCES.get(label);
}

function splitCitation(raw) {
  const labels = [];
  const semicolonParts = raw.split(";").map((part) => part.trim());
  let authorPrefix = "";
  for (const part of semicolonParts) {
    const multiYear = part.match(
      /^(.*?,\s*)((?:19|20)\d{2}[a-z]?)\s*,\s*((?:19|20)\d{2}[a-z]?)$/,
    );
    if (multiYear) {
      labels.push(
        `${multiYear[1]}${multiYear[2]}`,
        `${multiYear[1]}${multiYear[3]}`,
      );
      authorPrefix = multiYear[1];
      continue;
    }
    const full = part.match(/^(.*?,\s*)((?:19|20)\d{2}[a-z]?)$/);
    if (full) {
      authorPrefix = full[1];
      labels.push(`${authorPrefix}${full[2]}`);
      continue;
    }
    if (/^(?:19|20)\d{2}[a-z]?$/.test(part) && authorPrefix) {
      labels.push(`${authorPrefix}${part}`);
      continue;
    }
    labels.push(part);
  }
  return labels;
}

function linkify(raw, context, file, unknown) {
  return splitCitation(raw)
    .map((original) => {
      const label = NORMALIZE.get(original) ?? original;
      const url = sourceFor(label, context, file);
      if (!url) {
        unknown.add(`${file}: [${original}]`);
        return `[${original}]`;
      }
      return `[[${label}]](${url})`;
    })
    .join("; ");
}

function markdownFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if ([".vitepress", "plans"].includes(entry.name)) return [];
      return markdownFiles(full);
    }
    return entry.isFile() &&
      entry.name.endsWith(".md") &&
      entry.name !== "run-evidence.md"
      ? [full]
      : [];
  });
}

const authorStart = "(?:[A-ZÅ]|van\\s)";
const citationPattern = new RegExp(
  `(?<!\\[)\`?\\[(${authorStart}[^\\[\\]\\n]*?,\\s*(?:19|20)\\d{2}[a-z]?` +
    `(?:\\s*[,;]\\s*(?:(?:19|20)\\d{2}[a-z]?|${authorStart}[^\\[\\]\\n]*?,\\s*(?:19|20)\\d{2}[a-z]?))*)\\]\`?(?![\\](])`,
  "g",
);
const remainingPattern = new RegExp(
  `(?<!\\[)\`?\\[${authorStart}[^\\[\\]\\n]*?,\\s*(?:19|20)\\d{2}[a-z]?[^\\[\\]\\n]*?\\]\`?(?![\\](])`,
  "g",
);
const linkedCitationPattern =
  /\[\[([^\]]+?,\s*(?:19|20)\d{2}[a-z]?)\]\]\((https?:\/\/(?:[^()\s]|\([^()]*\))+?)\)/g;
const canonicalByUrl = new Map(
  [...SOURCES].map(([label, url]) => [url, label]),
);
const unknown = new Set();
let changed = 0;

for (const file of markdownFiles(DOCS)) {
  const relative = path.relative(ROOT, file);
  const input = fs.readFileSync(file, "utf8");
  const cleaned = FIX
    ? input
        .replace(/\[\[\[([^\]]+)\]\]\([^)]+\)\]\(([^)]+)\)/g, "[[$1]]($2)")
        .replace(linkedCitationPattern, (whole, label, url) => {
          const canonical = canonicalByUrl.get(url);
          return canonical ? `[[${canonical}]](${url})` : whole;
        })
    : input;
  const output = cleaned.replace(
    citationPattern,
    (whole, raw, offset, source) => {
      const start = Math.max(0, offset - 180);
      const end = Math.min(source.length, offset + whole.length + 180);
      return linkify(raw, source.slice(start, end), relative, unknown);
    },
  );
  if (FIX && output !== input) {
    fs.writeFileSync(file, output);
    changed += 1;
  }
}

const bare = [];
const mismatched = [];
let linkedCount = 0;
for (const file of markdownFiles(DOCS)) {
  const relative = path.relative(ROOT, file);
  const source = fs.readFileSync(file, "utf8");
  const lines = source.split("\n");
  lines.forEach((line, index) => {
    for (const match of line.matchAll(remainingPattern)) {
      bare.push(`${relative}:${index + 1}: ${match[0]}`);
    }
  });

  for (const match of source.matchAll(linkedCitationPattern)) {
    linkedCount += 1;
    const label = NORMALIZE.get(match[1]) ?? match[1];
    const start = Math.max(0, match.index - 180);
    const end = Math.min(source.length, match.index + match[0].length + 180);
    const expected = sourceFor(label, source.slice(start, end), relative);
    if (!expected) {
      unknown.add(`${relative}: [[${match[1]}]](${match[2]})`);
    } else if (label !== match[1]) {
      mismatched.push(
        `${relative}: [[${match[1]}]] is noncanonical; use [[${label}]]`,
      );
    } else if (match[2] !== expected) {
      mismatched.push(
        `${relative}: [[${match[1]}]] points to ${match[2]}; expected ${expected}`,
      );
    }
  }
}

if (unknown.size > 0 || bare.length > 0 || mismatched.length > 0) {
  if (unknown.size > 0) {
    console.error("Unknown citation keys:\n" + [...unknown].sort().join("\n"));
  }
  if (bare.length > 0) {
    console.error("Bare author-year citations:\n" + bare.join("\n"));
  }
  if (mismatched.length > 0) {
    console.error(
      "Citation labels pointing to the wrong source:\n" + mismatched.join("\n"),
    );
  }
  process.exitCode = 1;
} else {
  console.log(
    `Citation audit passed (${linkedCount} linked citations; ${changed} files updated).`,
  );
}

if (ONLINE && process.exitCode !== 1) {
  const citationLinkPattern =
    /\[\[[^\]]+?,\s*(?:19|20)\d{2}[a-z]?\]\]\((https?:\/\/(?:[^()\s]|\([^()]*\))+?)\)/g;
  const urls = new Set();
  for (const file of markdownFiles(DOCS)) {
    const source = fs.readFileSync(file, "utf8");
    for (const match of source.matchAll(citationLinkPattern))
      urls.add(match[1]);
  }

  const queue = [...urls].sort();
  const broken = [];
  const indeterminate = [];
  let cursor = 0;

  async function worker() {
    while (cursor < queue.length) {
      const url = queue[cursor++];
      try {
        const response = await fetch(url, {
          headers: { "User-Agent": "hands-on-world-models-citation-audit/1.0" },
          redirect: "follow",
          signal: AbortSignal.timeout(20000),
        });
        if ([404, 410].includes(response.status))
          broken.push(`${response.status} ${url}`);
        else if (response.status >= 400)
          indeterminate.push(`${response.status} ${url}`);
      } catch (error) {
        indeterminate.push(`${error.name} ${url}`);
      }
    }
  }

  await Promise.all(Array.from({ length: 10 }, () => worker()));
  console.log(`Checked ${queue.length} unique citation targets.`);
  if (indeterminate.length > 0) {
    console.warn(
      `Indeterminate responses (often bot protection or rate limiting):\n${indeterminate.join("\n")}`,
    );
  }
  if (broken.length > 0) {
    console.error(`Broken citation targets:\n${broken.join("\n")}`);
    process.exitCode = 1;
  }
}
