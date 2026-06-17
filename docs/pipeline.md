# Pipeline

The pipeline is intentionally split into simulation, analysis, and plotting.

```text
simulate -> data
analyze  -> statistics / rankings
plot     -> figures
```

Simulation code must not rely on figure generation. Plotting code must not run simulation.

## Phase Order

P0 generates a reusable surface bank.

P1 runs rigid and compliant single-case sanity checks and stores diagnostic arrays for manual inspection.

P2 screens compliant single-spine `spring_k_n_per_m x alpha_p_deg`.

P3 screens rigid single-spine `alpha_p_deg`.

P5a performs a coarse array size and pitch screen.

P5b refines P5a selected candidates and selects final rigid and compliant candidates.

P6 runs final Monte Carlo over P5b candidates.

P7 reads P6 data and computes surface generalization; it does not rerun simulation.

P8 reads P6 data and computes preload efficiency; it does not rerun simulation.

## Single Case Solver Order

The core case solver follows this order:

1. Load surface arrays from the surface bank.
2. Use `height_filtered`.
3. Interpolate spine heights and compute initial gaps.
4. Solve local normal preload `W_i` from `w_total_n`.
5. Compute critical engagement angles only after `W_i` exists.
   `trial_force_n` is interpreted as a per-spine trial tangential force, so
   lower-preload spines get a higher `phi_c` threshold instead of having the
   criterion collapse through nominal-count averaging.
6. Search finite tangential travel for the first engagement event.
7. Compute per-spine capacity with a finite upper bound.
   Capacity modes are `none`, `no_geometric_engagement`, `geom_friction`,
   `strength`, and `self_lock_strength`.
8. Run event-driven tangential loading and failure.
9. Save summary and per-spine rows.

The code explicitly records diagnostic order in P1/P3 single-case tests. Engagement must never be decided before preload.

## Failure Handling

Cases must preserve status information:

- `case_status`
- `error_code`
- `warning_flags`

Failed cases are recorded in manifests and tables; they should not be silently dropped.
