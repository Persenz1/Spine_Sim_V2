# Spine_Sim_V2

新版爪刺阵列仿真工程，用于“推力预载下硬质粗糙表面爪刺阵列准静态仿真”。工程目标是趋势比较、机制解释和结构筛选；未经实验标定，不用于真实墙面绝对承载预测。

## 模型边界

当前模型只面向硬质粗糙表面上的单爪或单爪单元。它不纳入完整对置爪夹持、整机姿态动力学、高速冲击、振动、惯性效应、表面刮削、颗粒脱落、粉化、磨损或刺尖磨钝；也不让微损伤直接提高承载力，不直接迁移文献材料参数，不把名义刺数当作有效刺数，不用 Ra 或 Rq 单独判断可啮合，不把自锁区数学发散解释成无限承载。

单个 case 的物理顺序固定为：

```text
读取 surface bank
-> 使用刺尖半径探针滤波后的 height_filtered
-> 计算初始间隙 g_i
-> 由 w_total_n 反解局部预载 W_i
-> 由 W_i 计算临界接合角 phi_c 和可接合区域
-> 沿切向有限行程搜索首次接合点
-> 计算单刺承载上限 F_t,i^cap
-> 执行切向位移控制和事件驱动失效
-> 输出 summary、spines、统计结果和可再生图片
```

必须先求局部预载 `W_i`，再判断可接合区域。没有 `W_i` 时不允许提前计算接合成功。
`trial_force_n` 在当前实现中定义为单刺临界接合试探力，默认取 `0.50 N`
（与 `F_ref_star_n` 同量级），不再按名义刺数摊薄；这样 `phi_c = atan(F_trial/W_i)-phi_s`
会保留局部预载对可接合区域的影响。

## 安装

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

快速检查：

```bash
python scripts/simulate.py --help
python scripts/analyze_results.py --help
python scripts/plot_results.py --help
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## 工程架构

工程采用数据优先、绘图解耦的三层流程：

```text
simulate -> data
analyze  -> statistics / rankings
plot     -> figures
```

`simulate` 只负责运行仿真并保存标准化数据。`analyze` 从已保存数据重新计算分组统计、评分、排名和候选清单。`plot` 只读取数据和分析结果生成图片，因此图片可以删除后重新生成，不需要重跑仿真。

## 性能、并行与内存

所有仿真阶段（P0/P2/P3/P5a/P5b/P6）都支持多进程并行，并自动按 CPU 线程数选择进程数：

- 命令行加 `--workers N` 显式指定进程数；省略或传 `0` 表示自动（取 `os.cpu_count()`）。
- 每个 case 的结果以批次（默认每 2000 行一个 Parquet row group）流式落盘，全程内存占用恒定，可应对百万级 Monte Carlo，不会把全部结果堆在内存里。
- 运行时在终端原位刷新单行进度条（显示完成比例、用时与预计剩余时间）；重定向到文件或非交互环境时只在结束打印一行汇总，不刷屏。

并行采用 `forkserver`/`spawn` 启动方式以规避多线程 `fork` 死锁。**若直接调用 `Spine_Sim_V2.pipelines.*` 的 `run_*` 函数（而非通过 `scripts/*.py`），驱动脚本必须放在 `if __name__ == "__main__":` 守卫内**；随包提供的 `scripts/simulate.py` 等入口已包含该守卫。

## 数据产品

surface bank 长期保存表面数组：

- `height_raw`：原始代理高度图。
- `height_filtered`：经过刺尖半径形态学闭运算探针滤波后的有效高度图；窄谷被填高，
  并保持 `h_eff >= h_raw` 的逐点单调性。`probe_filter_version` 为
  `morphological_closing_tip_v003` 的 surface bank 与旧版本下游结果不兼容，需要重建。
- `surface_statistics.parquet`：每个 `surface_id` 的统计信息。

case 表只保存 `surface_bank_id` 和 `surface_id`，不重复保存大规模地形数组。P1 的 `sample_cases/*/case_arrays.npz` 是人工诊断副本，不作为大规模阶段的默认输出。

主要文件格式：

- Parquet：权威表格格式，例如 `stage_summary.parquet`、`stage_spines.parquet`、`final_summary.parquet`。
- CSV：人工预览表，只保存前若干行，不作为权威数据。
- NPZ：数值数组，例如 surface bank 高度图和 P1 诊断数组。
- `manifest.json`：记录项目名、代码版本、模型版本、surface bank、随机种子策略、参数网格、期望/完成 case 数和失败 case。
- `schema.json`：记录表字段、dtype、单位、是否可空和说明。

`stage_grouped_statistics.parquet`、`final_grouped_statistics.parquet` 等分析派生表的
`schema.json` 会按实际 DataFrame 动态推断，避免统计列增加后 schema 描述滞后。

`data/`、`outputs/`、Parquet、NPZ 和生成图片默认不进入 Git，因为它们可能很大，并且可以由配置和代码复现。精选论文图片可手动复制到 `docs/figures/`。

## 最小流程

生成一个小型 surface bank：

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

审查 surface bank：

```bash
python scripts/plot_results.py surface-audit \
  --surface-bank data/surface_bank_debug \
  --sample-per-kind 3 \
  --outdir outputs/P0_surface_audit \
  --style report
```

运行 P1 单算例闭环：

```bash
python scripts/simulate.py p1-single-case \
  --surface-bank data/surface_bank_debug \
  --surface-id concrete_000000 \
  --outdir outputs/P1_single_case_sanity

python scripts/plot_results.py p1 \
  --stage-dir outputs/P1_single_case_sanity \
  --style debug
```

运行 P2/P3 单刺初筛：

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

运行 P5 阵列间距筛选：

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

运行 P6 最终 Monte Carlo 的 smoke 或正式接口：

```bash
python scripts/simulate.py p6-final-mc \
  --surface-bank data/surface_bank_debug \
  --selected-candidates outputs/P5b_array_pitch_refine_screen/data/selected_candidates.json \
  --n-surfaces-per-kind 3 \
  --surface-selection first_n \
  --outdir outputs/P6_final_3d_monte_carlo \
  --workers 1
```

正式运行可使用更大的表面库，例如 `data/surface_bank_v001` 和 `--n-surfaces-per-kind 1000`。同一接口也支持后续扩展到每类 1500 或 2000 个表面。

## 后处理

P7 和 P8 都从 P6 已保存数据后处理，不重新仿真：

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

重新生成图片：

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

可用绘图风格为 `debug`、`report` 和 `paper`，配置文件位于 `plot_styles/`。

## 文档

- [数据 Schema](docs/data_schema.md)
- [仿真流程](docs/pipeline.md)
- [Surface Bank](docs/surface_bank.md)
- [评分规则](docs/scoring.md)
- [绘图系统](docs/plotting.md)

## 目录结构

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
