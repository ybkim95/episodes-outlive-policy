# Episodes that outlive the policy

Two components of an async RL stack for long, multi-turn agent episodes, on a small testbed:

1. **Trajectory record** (`traj.py`, `runner.py`): every turn stores the policy version that sampled it, the exact context it saw, behaviour log-probs, and enough env state to fork the episode mid-way.
2. **Off-policy objectives** (`offpolicy.py`): PPO-clip, OAPL, CORO with a drift budget, and **Branched CORO** (`tree_loss`), which forks an episode at turn *b* into *k* continuations and contrasts suffixes among siblings. Per-turn drift accounting and two ways of enforcing a per-turn budget (both of which hurt; see below).

Staleness is controlled and heterogeneous *within* an episode. Sampler weights advance every `U` turns, and each episode also gets a heavy-tailed queue delay `q ≤ qmax`, so early turns are older than late ones.

## Layout

| file | what it is |
|---|---|
| `env.py` | `MemoryDialogue`: user, tool, teammate, memory slot, mid-episode goal change, reward only on the final reply, 0.01 cost for speaking early |
| `traj.py` | `Rollout` record and `build_context` (exact per-turn context) |
| `policy.py` | tiny causal transformer (d = 64, 2 layers); `act` and `score` over agent tokens only |
| `credit.py` | outcome advantages (GRPO-style and Dr.GRPO-style) for the `none`/`ppo` objectives |
| `offpolicy.py` | `loss` (none / ppo / oapl / coro), `tree_loss` (Branched CORO), `turn_kl_penalty`, `drift_stats` |
| `runner.py` | per-turn staleness, versioned rollout with forking, two-pass chunked backward, training loop, dual ascent |
| `tests/test_sanity.py` | 14 tests that must pass before any number is trusted |
| `analyze.py` | every table in the write-up, from `runs/*/metrics.jsonl` only |
| `talk_figs.py` | slide-sized figures (`figs/talk_*.png`) |
| `plots.py` | ad-hoc curves for any run glob |
| `scripts/` | the exact launch scripts (see History) |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests/                 # 14 tests, ~10 s on CPU
bash scripts/main.sh a                     # E0-E3 on seeds 0-2, drift-budget tuning on seed 100
bash scripts/main.sh b 0.002 1             # E4-E6 with the chosen budget
bash scripts/severity.sh                   # staleness up to 31 and 63
bash scripts/e7.sh                         # per-turn KL penalty (E7), three step sizes
python analyze.py                          # tables + figs/main_*.png, figs/severity.png, figs/e7_drift.png
python talk_figs.py                        # figs/talk_*.png
```

The scripts assume three GPUs (`CUDA_VISIBLE_DEVICES` 1–3). One run takes about 25–45 minutes on a shared GPU; the model is tiny, so CPU works too (`--device cpu`), just slower.

Protocol: every hyperparameter was chosen on seed 100 only, and every reported number is from seeds 0–2. Runs are matched on agent turns generated (≈4.1M each, branched runs included).

## Results (H = 16, full context, staleness 0–11 within episodes)

Output of `python analyze.py`. "Final" is the mean of the last 3 evals on 512 held-out episodes.

| config | final goal rate (seeds 0, 1, 2) | M turns to 0.9 |
|---|---|---|
| E0 CORO, no staleness | 0.52, 0.37, 1.00 | 1.85 (1 of 3 seeds) |
| E1 PPO-clip | 0.99, 1.00, 0.98 | 2.46 |
| E1b ignore staleness | 0.24, 0.20, 0.25 | never |
| E2 OAPL | 0.80, 1.00, 1.00 | 2.15 (2 of 3) |
| E3 CORO | 0.91, 1.00, 0.99 | 2.46 |
| E4 CORO + drift budget | 0.91, 0.99, 0.99 | 2.12 |
| **E5 Branched CORO + budget** | **1.00, 1.00, 1.00** | **1.31** |
| E6 Branched CORO + per-turn β | 0.73, 0.67, 0.98 | 3.35 (1 of 3) |
| E7 Branched CORO + per-turn KL penalty | 0.75, 0.78, 0.93 | 3.35 (1 of 3) |

- **Branched CORO** reaches 0.9 about 1.6× sooner than CORO with the same drift budget.
- **Staleness severity** (`severity.sh`, staleness up to 63): ignoring staleness falls to chance. PPO-clip, OAPL and CORO all average ≥ 0.93 at every level. In this testbed IS clipping did *not* become unstable. Drift stayed small: median trajectory ESS ≥ 0.71 of the batch, token ratios ≤ 6.3.
- **Per-turn budgets** hurt in both forms:
  - E6, a per-bucket β, also scales each turn's step size, so fresh, rewarded turns starve.
  - E7, a separate per-bucket KL penalty, does not starve those turns but still slows learning, more so with stronger penalties.
  - Both mostly fail on goal-change episodes.
  - Drift grows with staleness, so a uniform budget binds only on the oldest data.

## Limits

- One synthetic env, 16-turn episodes, full context visible. With context shorter than the episode, no method learned. Write-then-recall pays nothing until both halves work, so that is an exploration problem, not a credit one.
- A 64-wide, 2-layer policy; 32 agent tokens per episode. This is far from the token counts where IS ratios are reported to explode.
- 3 seeds per configuration. Failure analysis uses the 8 saved held-out episodes per run (`trajectories.txt`).

## History

The scripts record how the setup was reached. Each one's header says what it tested.

- `day1.sh`: first learnability check. At 64 turns with truncated context, the policy collapsed to one answer: an exploration problem.
- `day2*.sh`: env v2, the β sweep on seed 100 (0.03 → 0.89, 0.1 → 0.43, 0.3 → 1.00, 1.0 → 0.13), and the switch to full context. The first env v2 runs collapsed to always-PASS. Only the ANSWER verb was rewarded, and std-normalised advantages inflated the 0.01 speech cost. The final reply is now scored whatever its verb.
- `main.sh`, `severity.sh`, `e7.sh`: the reported experiments.

The run folders under `runs/` from these stages are kept, including the failed ones.
