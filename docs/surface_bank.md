# Surface Bank

The surface bank is the long-lived source of terrain data. Later stages reference surfaces by `surface_bank_id` and `surface_id`.

## Directory Layout

```text
data/surface_bank_v001/
  manifest.json
  schema.json
  surface_statistics.parquet
  surface_statistics_preview.csv
  surfaces/
    concrete_000000.npz
    sandpaper_000000.npz
```

Each NPZ contains:

- `height_raw`
- `height_filtered`

## Proxy Surface Kinds

The built-in proxy kinds are:

- `sandpaper`
- `concrete`
- `brick`
- `painted_wall`

Their roughness settings are proxy categories, not true material parameters and not absolute wall-load predictors.

## Resolution

Default resolution is 5 cells/mm, so `dx_mm = dy_mm = 0.2`.

High-fidelity display can use 10 cells/mm, so `dx_mm = dy_mm = 0.1`.

## Generation Model

The generator combines multiscale components:

```text
h = h0 + h_peak + h_pit + h_step + h_texture
```

The generated height map is mean-centered, normalized to target proxy `Rq`, saved as `height_raw`, and then passed through `probe_filter(height_raw, tip_radius_mm)` to produce `height_filtered`.
The current probe filter is `morphological_closing_tip_v003`: a flat disk gray-scale closing that fills valleys narrower than the tip footprint and preserves the pointwise bound `height_filtered >= height_raw`. Regenerate old banks before comparing downstream results.

## Rules

Do not save audit plots inside the surface bank directory. Use `outputs/P0_surface_audit` or another output path.

Do not duplicate surface arrays in large case outputs. Store only `surface_id`.

Use the surface statistics table for roughness and slope metadata.
