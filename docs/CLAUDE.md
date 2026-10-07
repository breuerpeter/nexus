`mkdocs` builds the site. `mkdocs.yml` sits at the repo root, and `uv run --group docs mkdocs serve|build` serves or builds it.

- `mkdocstrings` builds the pages under `reference/api/` from source docstrings. It uses griffe,
  which is **static**: it never imports or executes the Warp and Newton stack. Config is the
  `mkdocstrings` block of `mkdocs.yml`. The docs cover only the public `nexus_sim.*` surface.
- **3D vehicle previews.** Vehicle pages embed interactive `<model-viewer>` previews, vendored
  under `javascripts/`. The hook `hooks/vehicle_previews.py` generates them from
  `nexus_sim/_src/config/catalog.yaml` wherever it finds a `<!-- model-preview: <name> -->`
  marker. `<name>` is the Universal Scene Description (USD) filename stem, for example
  `astro_max_fpv`. The `.glb` is the content-addressed sibling of the vehicle's USD on
  CloudFront. When a local copy exists under `assets/local/` at the repo root, the site serves
  that copy instead: a preview before publishing, injected into the site without writing into
  `docs/`.
- **Benchmark trends.** `javascripts/bench-trends.js` draws the Benchmarking page's trend panels
  in the browser from the CI bucket's bench feed, which the bucket serves to the deployed origin
  only. `hooks/benchmarks.py` only states which metrics to plot and their gates, from
  `scripts/ci/examples_baselines.json`, at the `<!-- benchmark-trends -->` marker. A local
  `mkdocs serve` shows a note in place of the panels, which isn't a fault.
- Headings are sentence case. `scripts/check_heading_case.py`, a pre-commit hook, checks them.
- `exclude_docs` in `mkdocs.yml` keeps this file out of the site. Every other `.md` here is a page.
