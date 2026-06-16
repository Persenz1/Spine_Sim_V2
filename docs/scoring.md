# Scoring

Scoring is used for screening and ranking candidates. It is not a calibrated physical load predictor.

## Normalization

Metrics are normalized by Q05-Q95.

For larger-is-better metrics:

```text
S_high(x) = clip((x - Q05) / (Q95 - Q05), 0, 1)
```

For smaller-is-better metrics:

```text
S_low(x) = 1 - clip((x - Q05) / (Q95 - Q05), 0, 1)
```

If `Q95 == Q05`, all candidates receive score `0.5` for that metric.

## P2

Candidate unit: `spring_k_n_per_m x alpha_p_deg`.

The score combines success probability, force, efficiency, surface robustness, low search failure, and low saturation. Surface robustness uses the p25 of success probability across surface kinds.

P2 keeps 6 to 8 diverse candidates when available. Diversity rules avoid selecting only one angle or only the maximum stiffness.

## P3

Candidate unit: `alpha_p_deg`.

P3 keeps 60 degrees as a baseline and one or two additional strong angles.

## P5

Candidate unit: array geometry plus rigidity/compliance parameters.

The P5 score combines:

- success probability
- force
- efficiency
- surface robustness
- preload robustness
- Kish effective count
- low `eta_max`
- low search failure
- low saturation
- low failure diagnostics

P5b selects rigid and compliant finalists by roles such as best overall, highest force, highest success rate, balanced load sharing, and baseline/mid-stiffness references. Duplicate selections are skipped.

## P6

P6 final rankings reuse the P5-style metrics over a larger Monte Carlo sample. P6 output is intended for final comparison among P5b candidates, not for absolute wall prediction.
