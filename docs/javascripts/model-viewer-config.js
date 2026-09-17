// Configure <model-viewer> for the asset previews (docs/vehicles.md) and make failures
// visible instead of leaving a blank box.
//
// 1. Point the Draco/KTX2 decoders at the self-hosted copies (the glb is Draco-
//    compressed; the default gstatic.com CDN may be unreachable on restricted networks).
//    `import.meta.url` keeps the path correct under the site base prefix (/nexus/).
// 2. Overlay a status note per viewer: a WebGL-missing warning, a "loading" hint, or the
//    actual error — so a non-rendering preview explains itself.

customElements.whenDefined("model-viewer").then(() => {
  const ModelViewer = customElements.get("model-viewer");
  if (!ModelViewer) return;
  const here = (rel) => new URL(rel, import.meta.url).href;
  ModelViewer.dracoDecoderLocation = here("./draco/");
  ModelViewer.ktx2TranscoderLocation = here("./draco/");
});

function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

function annotate(mv) {
  const note = document.createElement("div");
  note.style.cssText =
    "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;" +
    "font:14px/1.4 system-ui,sans-serif;color:#555;text-align:center;padding:1em;pointer-events:none;";
  const host = mv.parentElement || mv;
  if (getComputedStyle(host).position === "static") host.style.position = "relative";
  host.appendChild(note);

  if (!hasWebGL()) {
    note.textContent = "This 3D preview needs WebGL, which isn't available in this browser.";
    return;
  }
  note.textContent = "Loading 3D model…";

  // model-viewer 4.x doesn't start loading `src` when the element is upgraded AFTER it
  // was parsed (our module script is deferred, so the element exists first). Re-asserting
  // `src` once the definition is ready kicks off the fetch. Without this the preview sits
  // at "Loading…" forever and never even requests the glb.
  customElements.whenDefined("model-viewer").then(() => {
    const src = mv.getAttribute("src");
    if (src && !mv.loaded) {
      mv.removeAttribute("src");
      requestAnimationFrame(() => mv.setAttribute("src", src));
    }
  });

  // `progress` reports fetch+decode 0→1; watching where it stalls tells us whether the
  // download or the Draco decode is the bottleneck.
  let pct = 0;
  mv.addEventListener("progress", (e) => {
    pct = Math.round((e.detail?.totalProgress ?? 0) * 100);
    note.textContent = pct < 100 ? `Loading 3D model… ${pct}%` : "Decoding 3D model…";
  });
  mv.addEventListener("load", () => note.remove(), { once: true });
  mv.addEventListener("error", (e) => {
    const d = e.detail || {};
    note.textContent =
      "Couldn't load the 3D model: " + (d.sourceError?.message || d.type || "unknown error");
  });
  // Surface a silent stall: if nothing has loaded after 30s, report the last %.
  setTimeout(() => {
    if (note.isConnected && !note.textContent.startsWith("Decoding")) {
      note.textContent = `Still loading at ${pct}% after 30s — likely a network/decode stall.`;
    }
  }, 30000);
}

if (document.readyState !== "loading") {
  document.querySelectorAll("model-viewer").forEach(annotate);
} else {
  document.addEventListener("DOMContentLoaded", () =>
    document.querySelectorAll("model-viewer").forEach(annotate)
  );
}
