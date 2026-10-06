// Draw the benchmark trend panels on the Benchmarking page from the per-commit bench feed.
//
// The build leaves a `.bench-trends` block (docs/hooks/benchmarks.py) that names the feed's index
// and, as JSON, each gated metric with the bound its gate enforces. This script fetches the index,
// keeps the newest commits, fetches each one's `bench/<sha12>.json` and draws one small SVG panel
// per gated metric under its example: the values in commit order, the gate as a dashed line, a
// point's commit, date and value on hover, and the commit's page on click. A metric no commit
// measured has no panel. A per-commit file that fails to load leaves the other commits plotted.
// An index the browser cannot fetch, as from an origin the bucket's CORS allowlist does not name,
// leaves a message in place of the panels.

const NEWEST = 50; // commits plotted, so a long history stays one fetch per point on the page
const WIDTH = 240;
const HEIGHT = 96;
const PAD = { left: 44, right: 10, top: 8, bottom: 8 };
const SVG = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text = "") {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  if (text) node.textContent = text;
  return node;
}

function fmt(value) {
  return Number.isInteger(value) ? String(value) : value.toPrecision(3);
}

async function feed(feedUrl) {
  const indexUrl = new URL(feedUrl, location.href); // a key resolves against the bucket the index came from
  const index = await (await fetch(indexUrl)).json();
  const listed = index
    .filter((record) => record && typeof record.key === "string")
    .sort((a, b) => String(a.date).localeCompare(String(b.date)))
    .slice(-NEWEST);
  const loads = listed.map(async (record) => {
    const entries = await (await fetch(new URL("/" + record.key, indexUrl))).json();
    const sha = record.key.split("/").pop().replace(/\.json$/, "");
    const values = new Map(entries.map((entry) => [entry.name, entry]));
    return { sha, date: record.date, values };
  });
  return (await Promise.allSettled(loads)).filter((r) => r.status === "fulfilled").map((r) => r.value);
}

// One panel: the metric's values over `commits`, those that measured it, on a y scale that holds
// every value and the gate. Consecutive commits join with a line; a commit without the metric
// breaks it, so a gap stays a gap.
function panel(name, bound, commits, repo) {
  const measured = commits.map((commit) => commit.values.get(name));
  const values = measured.filter(Boolean).map((entry) => entry.value);
  const span = [...values, ...Object.values(bound)];
  let lo = Math.min(...span);
  let hi = Math.max(...span);
  if (hi === lo) (lo -= 1), (hi += 1);
  const margin = (hi - lo) * 0.1;
  (lo -= margin), (hi += margin);
  const inset = 6; // keeps the first and last points off the gate labels and the edge
  const x = (i) =>
    PAD.left + inset + (commits.length < 2 ? 0.5 : i / (commits.length - 1)) * (WIDTH - PAD.left - PAD.right - 2 * inset);
  const y = (v) => PAD.top + (1 - (v - lo) / (hi - lo)) * (HEIGHT - PAD.top - PAD.bottom);

  const svg = el("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, class: "bench-panel", role: "img" });
  for (const edge of ["min", "max"]) {
    if (!(edge in bound)) continue;
    const gy = y(bound[edge]);
    svg.append(el("line", { x1: PAD.left, x2: WIDTH - PAD.right, y1: gy, y2: gy, class: "bench-gate" }));
    svg.append(el("text", { x: PAD.left - 4, y: gy + 3, class: "bench-label", "text-anchor": "end" }, fmt(bound[edge])));
  }
  // One `M … L …` run per stretch of consecutive measured commits; a missing one ends the run.
  const runs = [];
  measured.forEach((entry, i) => {
    if (!entry) return runs.push([]);
    if (!runs.length) runs.push([]);
    runs.at(-1).push(`${x(i)} ${y(entry.value)}`);
  });
  const path = runs.filter((run) => run.length).map((run) => "M " + run.join(" L ")).join(" ");
  svg.append(el("path", { d: path, class: "bench-line" }));
  measured.forEach((entry, i) => {
    if (!entry) return;
    const commit = commits[i];
    const link = el("a", { href: `${repo}/commit/${commit.sha}` });
    const dot = el("circle", { cx: x(i), cy: y(entry.value), r: 3, class: "bench-point" });
    dot.append(el("title", {}, `${commit.sha} · ${commit.date} · ${fmt(entry.value)} ${entry.unit || ""}`.trim()));
    link.append(dot);
    svg.append(link);
  });
  const last = measured.filter(Boolean).at(-1);
  const figure = document.createElement("figure");
  figure.className = "bench-figure";
  const caption = document.createElement("figcaption");
  caption.textContent = `${name.split("[")[0]} · ${fmt(last.value)} ${last.unit || ""}`.trim();
  figure.append(caption, svg);
  return figure;
}

async function draw(block) {
  const gates = JSON.parse(document.getElementById("bench-gates").textContent);
  const status = block.querySelector(".bench-trends-status");
  let commits;
  try {
    commits = await feed(block.dataset.feed);
  } catch {
    status.textContent = "The trend feed is not reachable from this origin.";
    return;
  }
  const byExample = new Map();
  for (const [name, bound] of Object.entries(gates)) {
    if (!commits.some((commit) => commit.values.has(name))) continue;
    const example = name.slice(name.indexOf("[") + 1, -1);
    if (!byExample.has(example)) byExample.set(example, []);
    byExample.get(example).push(panel(name, bound, commits, block.dataset.repo));
  }
  status.remove();
  for (const [example, panels] of byExample) {
    const heading = document.createElement("h4");
    heading.textContent = example;
    const row = document.createElement("div");
    row.className = "bench-row";
    row.append(...panels);
    block.append(heading, row);
  }
  const note = document.createElement("p");
  note.innerHTML = `<small>${commits.length} commits, newest last. Dashed: the gate. Hover a point for its commit; click to open it.</small>`;
  block.append(note);
}

document.querySelectorAll(".bench-trends").forEach(draw);
