# Reviewer response — label noise robustness

We resample each guidance label uniformly at random with probability p, applied once when the
label enters the rollout buffer, and run 10 seeds per cell on DynamicCrossing and FragileCrossing.

| Env | p=0.0 | p=0.1 | p=0.25 | p=0.5 | p=1.0 |
|---|---|---|---|---|---|
| DynamicCrossing — Reward | 1.53 ± 0.01 | 1.53 ± 0.01 | 1.53 ± 0.01 | 1.53 ± 0.01 | 1.53 ± 0.01 |
| DynamicCrossing — Cost | 0.00 ± 0.00 | 0.01 ± 0.02 | 0.00 ± 0.00 | 0.75 ± 0.07 | 0.82 ± 0.05 |
| FragileCrossing — Reward | 1.02 ± 0.17 | 1.02 ± 0.17 | 1.06 ± 0.12 | 1.08 ± 0.06 | 0.97 ± 0.07 |
| FragileCrossing — Cost | 0.02 ± 0.04 | 0.02 ± 0.04 | 0.04 ± 0.11 | 0.17 ± 0.54 | 1.60 ± 0.84 |

The cost limit is 0.1 in both environments (Table 4 of the paper), so p=0.5 and p=1.0 are
infeasible in both, while p<=0.25 is feasible in both.

The results show a threshold effect. Below it, Split-RL is robust to label noise: up to p=0.25
both reward and cost are unchanged from the clean run in both environments, so the method does
not require precise guidance labels. Above it, the noise degrades the policy — cost rises from
0.00 to 0.82 in DynamicCrossing and from 0.02 to 1.60 in FragileCrossing.

---

## Provenance

- Source: `results/csv/minigrid_df_rebuttal_ablation_noise.csv`, aggregated as in
  `results/minigrid_ablation_rebuttle_noise.ipynb` (last logged `global_step` per seed,
  then mean ± std across seeds).
- Seeds: `[0, 5, 10, 42, 64, 100, 101, 102, 103, 104]`, 10 per cell in both environments.
- Noise implementation: `apply_label_noise` in `algos/safety/label_ablation.py`, called once at
  rollout storage in `algos/split_rl.py` (not in the training loop, so the corruption does not
  average out across epochs).
- Corner is excluded — runs incomplete at the time of writing.
