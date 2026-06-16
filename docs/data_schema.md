# Data Schema

This project saves tabular data as Parquet, preview tables as CSV, arrays as NPZ, and reproducibility metadata as JSON.

## Required Metadata

Every generated stage directory contains:

- `manifest.json`: project name, created time, code version, model version, schema version, surface bank id, random seed policy, parameter grid, expected/completed case counts, failed cases, and notes.
- `schema.json`: field name, dtype, unit, nullability, and description for saved tables.

## Surface Bank Tables

`surface_statistics.parquet` records one row per `surface_id`.

Important fields include:

- `surface_bank_id`
- `surface_id`
- `surface_kind`
- `seed`
- `dx_mm`, `dy_mm`
- `size_x_mm`, `size_y_mm`
- `tip_radius_mm`
- raw/effective roughness statistics
- slope statistics
- `candidate_density_preload_free`
- `valid`
- `reject_reason`

Surface arrays are not embedded in this table. They live in `surfaces/{surface_id}.npz`.

## Case Summary Tables

Screening stages use `stage_summary.parquet`; P6 uses `final_summary.parquet`.

Summary rows contain:

- status fields: `case_status`, `error_code`, `warning_flags`
- surface references: `surface_bank_id`, `surface_id`, `surface_kind`
- candidate and geometry fields with units in names
- preload and search settings
- surface statistics copied from the surface bank metadata
- contact, engagement, effective count, and load metrics
- success fields: `engagement_success` and `load_success`
- failure diagnostics

Rigid arrays must store `spring_k_n_per_m = null` and `spring_k_n_per_mm = null`; never use `0`.

## Spine Tables

Screening stages use `stage_spines.parquet`; P6 uses `final_spines.parquet`.

Each row is one spine and includes:

- `case_id`, `candidate_id`, `surface_id`
- `spine_id`, `row`, `col`, `x_mm`, `y_mm`
- gap and preload fields
- contact/engagement state
- engagement coordinates and angles
- capacity and failure fields

No spine table stores raw or filtered height maps.

## Arrays

Surface bank NPZ files store only:

- `height_raw`
- `height_filtered`

P1 diagnostic sample cases may additionally save arrays such as slope maps, search paths, engagement points, and load-displacement curves. Large screens do not save full arrays per case.
