# Plotting

Figures are generated only from saved data products.

## Entry Points

```bash
python scripts/plot_results.py surface-audit ...
python scripts/plot_results.py p1 ...
python scripts/plot_results.py stage ...
python scripts/plot_results.py final ...
python scripts/plot_results.py p7 ...
python scripts/plot_results.py p8 ...
```

`stage` covers P2, P3, P5, and P9 sensitivity tables when present. `final` covers P6.

## Styles

Plot styles live in `plot_styles/`:

- `debug.yaml`
- `report.yaml`
- `paper.yaml`

Style fields include:

- `figure_size`
- `dpi`
- `font_family`
- `font_size`
- `line_width`
- `marker_size`
- `colormap`
- `language`
- `save_format`

`debug` and `report` default to PNG. `paper` defaults to PDF. Curated figures can be copied manually into `docs/figures/`.

## Default Figures

P0 produces surface galleries and audit distributions.

P1 produces height, angle, hotspot, search path, spine state, and load-displacement debug figures.

P2/P3 produce screening heatmaps and curves.

P5 produces array heatmaps, ranking, and selected-candidate overview.

P6 produces final force, success, surface generalization, preload efficiency, rigid-vs-compliant, and convergence figures.

P7 produces surface generalization plots.

P8 produces preload force, efficiency, and success curves.

P9 sensitivity plotting is available when a saved `p9_sensitivity_statistics.parquet` or `sensitivity_statistics.parquet` exists.

## Rules

The plot layer never runs simulation. If figures are deleted, rerun `plot_results.py` from the saved data. Missing required columns raise explicit errors.
