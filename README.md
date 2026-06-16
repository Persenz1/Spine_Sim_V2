# Spine_Sim_V2

新版爪刺阵列仿真工程，用于“推力预载下硬质粗糙表面爪刺阵列准静态仿真”。工程目标是趋势比较、机制解释和结构筛选；未经实验标定，不用于真实墙面绝对承载预测。

## Model Boundary

当前模型只面向硬质粗糙表面上的单爪或单爪单元。它不纳入完整对置爪夹持、整机姿态动力学、高速冲击、振动、惯性效应、表面刮削、颗粒脱落、粉化、磨损或刺尖磨钝；也不让微损伤直接提高承载力，不直接迁移文献材料参数，不把名义刺数当作有效刺数，不用 Ra 或 Rq 单独判断可啮合，不把自锁区数学发散解释成无限承载。

单个 case 的物理顺序固定为：读取 surface bank，使用刺尖半径探针滤波后的有效高度图，计算初始间隙，由 `w_total_n` 反解局部预载 `W_i`，先得到 `W_i` 再计算临界接合角和可接合区域，沿切向有限行程搜索首次接合点，计算单刺承载上限，最后进行切向位移控制和事件驱动失效。

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Quick checks:

```bash
python scripts/simulate.py --help
python scripts/analyze_results.py --help
python scripts/plot_results.py --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Architecture

The project uses a data-first, plot-decoupled workflow:

```text
simulate -> data
analyze  -> statistics / rankings
plot     -> figures
```

Simulation scripts save standardized data only. Analysis scripts recompute statistics, rankings, and reports from saved data. Plot scripts only read saved data and analysis outputs, so figures can be deleted and regenerated without rerunning simulation.

## Data Products

Surface banks store long-lived terrain arrays:

- `height_raw` and `height_filtered` are stored once per `surface_id` in NPZ files.
- Case tables store `surface_bank_id` and `surface_id`; they do not duplicate terrain arrays.
- P1 sample cases may save diagnostic arrays for manual inspection.

Stage tables use Parquet as the authoritative format. CSV files are preview only. NPZ stores numerical arrays. `manifest.json` records reproducibility metadata, parameter grids, expected/completed case counts, and failures. `schema.json` records table fields, dtype, units, nullability, and descriptions.

`data/`, `outputs/`, Parquet, NPZ, and generated figure files are ignored by Git because they can be large and are reproducible from configuration and code. Curated paper figures may be copied manually into `docs/figures/`, which is explicitly allowed by `.gitignore`.

## Minimal Workflow

Generate a small surface bank:

```bash
python scripts/simulate.py p0-surface-bank \
  --bank-id surface_bank_debug \
  --surfaces sandpaper,concrete,brick,painted_wall \
  --n-per-kind 3 \
  --resolution 5 \
  --size-x-mm 24 \
  --size-y-mm 18 \
  --tip-radius-mm 0.05 \
  --outdir data/surface_bank_debug
```

Audit the surface bank:

```bash
python scripts/plot_results.py surface-audit \
  --surface-bank data/surface_bank_debug \
  --sample-per-kind 3 \
  --outdir outputs/P0_surface_audit \
  --style report
```

Run P1 single-case sanity:

```bash
python scripts/simulate.py p1-single-case \
  --surface-bank data/surface_bank_debug \
  --surface-id concrete_000000 \
  --outdir outputs/P1_single_case_sanity

python scripts/plot_results.py p1 \
  --stage-dir outputs/P1_single_case_sanity \
  --style debug
```

Run P2/P3 initial screens:

```bash
python scripts/simulate.py p2-compliant-k-alpha \
  --surface-bank data/surface_bank_debug \
  --n-surfaces-per-kind 3 \
  --outdir outputs/P2_compliant_k_alpha_screen

python scripts/simulate.py p3-rigid-alpha \
  --surface-bank data/surface_bank_debug \
  --n-surfaces-per-kind 3 \
  --outdir outputs/P3_rigid_alpha_screen

python scripts/analyze_results.py stage \
  --stage-dir outputs/P2_compliant_k_alpha_screen

python scripts/plot_results.py stage \
  --stage-dir outputs/P2_compliant_k_alpha_screen \
  --style report
```

Run P5 array pitch screens:

```bash
python scripts/simulate.py p5a-array-coarse \
  --surface-bank data/surface_bank_debug \
  --p2-selected outputs/P2_compliant_k_alpha_screen/data/selected_candidates.json \
  --p3-selected outputs/P3_rigid_alpha_screen/data/selected_candidates.json \
  --n-surfaces-per-kind 3 \
  --outdir outputs/P5a_array_pitch_coarse_screen

python scripts/simulate.py p5b-array-refine \
  --surface-bank data/surface_bank_debug \
  --p5a-selected outputs/P5a_array_pitch_coarse_screen/data/selected_candidates.json \
  --n-surfaces-per-kind 3 \
  --outdir outputs/P5b_array_pitch_refine_screen
```

Run P6 final Monte Carlo smoke or formal interface:

```bash
python scripts/simulate.py p6-final-mc \
  --surface-bank data/surface_bank_debug \
  --selected-candidates outputs/P5b_array_pitch_refine_screen/data/selected_candidates.json \
  --n-surfaces-per-kind 3 \
  --surface-selection first_n \
  --outdir outputs/P6_final_3d_monte_carlo \
  --workers 1
```

For a formal run, use a larger bank such as `data/surface_bank_v001` and `--n-surfaces-per-kind 1000`. The same command supports later expansion to 1500 or 2000 surfaces per kind.

Run P7/P8 post-processing from P6 data without rerunning simulation:

```bash
python scripts/analyze_results.py final \
  --stage-dir outputs/P6_final_3d_monte_carlo

python scripts/analyze_results.py p7-surface \
  --p6-dir outputs/P6_final_3d_monte_carlo \
  --outdir outputs/P7_surface_generalization

python scripts/analyze_results.py p8-preload \
  --p6-dir outputs/P6_final_3d_monte_carlo \
  --outdir outputs/P8_preload_efficiency
```

Regenerate figures from saved data:

```bash
python scripts/plot_results.py final \
  --stage-dir outputs/P6_final_3d_monte_carlo \
  --style report

python scripts/plot_results.py p7 \
  --stage-dir outputs/P7_surface_generalization \
  --style report

python scripts/plot_results.py p8 \
  --stage-dir outputs/P8_preload_efficiency \
  --style paper
```

Available plot styles are `debug`, `report`, and `paper`, configured in `plot_styles/`.

## Documentation

- [Data Schema](docs/data_schema.md)
- [Pipeline](docs/pipeline.md)
- [Surface Bank](docs/surface_bank.md)
- [Scoring](docs/scoring.md)
- [Plotting](docs/plotting.md)

## Package Layout

```text
Spine_Sim_V2/
  config/
  core/
  surfaces/
  solvers/
  pipelines/
  analysis/
  plotting/
  io/
  tests/
scripts/
plot_styles/
data/
outputs/
docs/
```
