# Episodes that outlive the policy

Two components of an async RL stack for long, multi-turn agent episodes, on a small testbed:

1. **Trajectory record** (`traj.py`, `runner.py`): every turn stores the policy version that sampled it, the exact context it saw, behaviour log-probs, and enough env state to fork the episode mid-way.
2. **Off-policy objectives** (`offpolicy.py`): PPO-clip, OAPL, CORO with a drift budget, and **Branched CORO** (`tree_loss`), which forks an episode at turn *b* into *k* continuations and contrasts suffixes among siblings. Per-turn drift accounting and two ways of enforcing a per-turn budget (both of which hurt; see below).

Staleness is controlled and differs *within* an episode. Sampler weights advance every `U` turns, and each episode also gets a heavy-tailed queue delay `q ≤ qmax`, so early turns are older than late ones.

The sampler and trainer are not separate processes. Past trainer weights sit in a ring buffer, and each turn is sampled with the weights its staleness points to. Staleness is therefore exact, and every run repeats bit for bit. A real system would replace the ring buffer with a queue between workers; the record and the losses would stay the same.

## In a real async stack (designed, not built)

The ring buffer stands in for a queue between sampler and trainer workers. What the record would have to handle there:

| issue | what goes wrong | design |
|---|---|---|
| weights change mid-episode | the sampler's KV cache was built by older weights | swap at turn boundaries; re-prefill or keep the old cache, but always log the log-probs the sampler actually computed, which are μ. This also absorbs numeric mismatch between the inference engine and the trainer, so fresh turns are never exactly on-policy |
| long episodes are staler | a cutoff on episode age drops long episodes first | bound the age of the last turn; per-turn versions keep the older turns usable |
| branch groups wait on stragglers | level 2 needs all k siblings, level 1 all trunks | take partial groups at a deadline: any group of 2 or more still centres |
| context storage | storing each turn's full context grows with the square of the turns | per-turn deltas plus a prefix pointer; forks share the prefix and its KV cache |

The ring buffer re-scores every turn with that turn's weights, which is the re-prefill option. `Rollout` currently stores full per-turn contexts.

## Layout

| file | what it is |
|---|---|
| `env.py` | `MemoryDialogue`: user, tool, teammate, memory slot, mid-episode goal change, reward only on the final reply, 0.01 cost for speaking early |
| `traj.py` | `Rollout` record and `build_context` (exact per-turn context) |
| `policy.py` | tiny causal transformer (d = 64, 2 layers); `act` and `score` over agent tokens only |
| `credit.py` | outcome advantages (GRPO-style and Dr.GRPO-style) for the `none`/`ppo` objectives |
| `offpolicy.py` | `loss` (none / ppo / oapl / coro), `tree_loss` (Branched CORO), `turn_kl_penalty`, `drift_stats` |
| `runner.py` | per-turn staleness, versioned rollout with forking, two-pass chunked backward, training loop, dual ascent, the control modes (`tree_flat`, `--beta_path`, `--tree_no_level1`, `--fork_resample`, `--crn`) |
| `tests/test_sanity.py` | 22 tests that must pass before any number is trusted |
| `analyze.py` | every table in the write-up (main, severity, E7, full-run drift, controls, follow-ups), from `runs/*/metrics.jsonl` only |
| `talk_figs.py` | slide-sized figures (`figs/talk_*.png`) |
| `plots.py` | ad-hoc curves for any run glob |
| `scripts/` | the exact launch scripts (see History) |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests/                 # 22 tests, ~20 s on CPU
bash scripts/main.sh a                     # E0-E3 on seeds 0-2, drift-budget tuning on seed 100
bash scripts/main.sh b 0.002 1             # E4-E6 with the chosen budget
bash scripts/severity.sh                   # staleness up to 31 and 63
bash scripts/e7.sh                         # per-turn KL penalty (E7), three step sizes
bash scripts/controls.sh                   # controls C1-C5 and reruns of E4/E5 (needs the E5 and E7 runs)
bash scripts/followup.sh                   # fixed-beta sweep, and E4/C1/E5 at 32 turns (about 1 h at 32 turns)
bash scripts/mechanism.sh                  # M1-M3: where Branched CORO's speed comes from
python analyze.py                          # tables + figs/main_*.png, severity.png, e7_drift.png, controls.png, beta_sweep.png
python talk_figs.py                        # figs/talk_*.png
```

The scripts assume three GPUs (`CUDA_VISIBLE_DEVICES` 1–3). One run takes about 25–45 minutes on a shared GPU; the model is tiny, so CPU works too (`--device cpu`), just slower.

Protocol: β, the learning rate (1e-3 over 3e-3) and the drift budget were chosen on seed 100 only, and every reported number is from seeds 0–2. The clip range (0.2), OAPL's β₁ (1), and the branch sizes (k = 4 continuations, 4 trunks per seed) are defaults I never tuned. Runs are matched on agent turns generated (≈4.1M each, branched runs included), except C2, which uses 1.6× the turns per update by design.

Trainer cost is not matched. A branched batch has 416 episodes against 256, and the trainer scores every turn of every episode, so it scores 1.6× the contexts per update. Measured in trainer compute, E4 and E5 reach 0.9 at about the same point; the saving is in sampled turns, which the prompt names as the expensive part. C1 and C2 have E5's trainer cost per update. Shared prefixes are scored k times; scoring them once would bring a branched update to about 3.8k contexts against 4.1k for E4 (not implemented).

Provenance: each run's `config.json` has `src_hash`. Until the controls it hashed every `.py` file, analysis scripts included, so it changed between stages without any training change. It now hashes only the six training files. E4 and E5 rerun on the final code match the original runs' loss bit for bit for all 1000 updates, seeds 0–2.

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

- Branched CORO reaches 0.9 about 1.6× sooner than CORO with the same drift budget. The controls below trace the gain to the sibling contrast (level 2): on the same forked data a flat loss needs 1.5× the updates, and level 2 alone is as fast as the full loss.
- Up to staleness 63 (`severity.sh`), ignoring staleness falls to chance, while PPO-clip, OAPL and CORO all average 0.93 or more at every level. IS clipping did not become unstable in this testbed. At staleness ≤ 63, all three seeds that ignore staleness collapsed to one fixed action per turn (for example `PASS G6`), so every ratio is 1, their trajectory ESS reads 1.00, and the final reply is one fixed goal.
- Drift stayed small, with rare spikes. Over every update of every run, the median trajectory ESS is at least 0.77 of the batch for every corrected method, and the 5th percentile at least 0.59. The largest token ratio per update has a median of at most 1.55, but its maximum reaches 15–214 depending on method and staleness (full-run drift table in `python analyze.py`).
- Both ways of enforcing a per-turn budget hurt, and both mostly fail on goal-change episodes.
  - E6 gives each staleness bucket its own β. A turn's β also scales its gradient. Fresh turns have no drift, so their β fell to the floor, and the rewarded final turn (in a fresh bucket 79% of the time) stopped learning.
  - E7 keeps one β and adds a KL penalty per bucket. The penalty kept KL under budget, so the dual pushed β to its floor by update ~430 while the run was still learning. The controls and the β sweep below show why that hurts.

## Controls (`scripts/controls.sh`)

| config | data per update | final goal (seeds 0, 1, 2) | updates to 0.9 (per seed) | M turns to 0.9 |
|---|---|---|---|---|
| E4 CORO + budget | 256 episodes | 0.91, 0.99, 0.99 | 651, 501, 401 | 2.12 |
| C2 CORO + budget | 416 episodes, no forking | 0.99, 1.00, 1.00 | 401, 501, 451 | 3.00 |
| C1 flat CORO on forked data | 104 trunks × 4 leaves | 0.91, 1.00, 1.00 | 651, 401, 401 | 2.00 |
| E5 Branched CORO | 104 trunks × 4 leaves | 1.00, 1.00, 1.00 | 351, 301, 301 | 1.31 |
| E7 (penalty step 100) | as E5 | 0.75, 0.78, 0.93 | –, –, 801 | – |
| C3 E7 with E5's β path | as E5 | 0.96, 0.98, 0.99 | 851, 351, 401 | 2.20 |
| C4 E5 with E7's β path | as E5 | 0.99, 1.00, 1.00 | 351, 301, 301 | 1.31 |
| E0 sync CORO | 256 episodes | 0.52, 0.37, 1.00 | –, –, 451 | – |
| C5 sync policy gradient (= sync PPO) | 256 episodes | 0.53, 0.34, 0.86 | –, –, – | – |

Evals are every 50 updates, so "updates to 0.9" has that resolution. "–" means the seed never reached 0.9.

- On the same forked data, flat CORO (C1) needs 484 updates on average and Branched CORO (E5) needs 318. More independent episodes per update (C2) help per update (451) but cost 1.6× the turns. Every E5 seed is faster than every seed of E4, C1 and C2. With 3 seeds per side, a complete separation has chance probability 1/20 per comparison. So the speed comes from the loss, not the data; the mechanism controls below narrow it down to level 2.
- Replaying E5's β path into E7 (C3) brings it to 0.96–0.99, though seed 0 is still slow. E7's own β reached the 1e-3 floor at updates 402–431. E5's stayed between 0.18 and 0.33 until E5 reached 0.9.
- C4 left E5 unchanged. I first read that as "β is inert under Adam", and that was wrong. Before E5 reached 0.9, the two β paths differed by at most 2–3×, so C4 only shows that E5 tolerates that much. The fixed-β sweep below is the real test.
- Sync is slow for every objective here: sync policy gradient (C5) matches sync CORO (E0). One explanation fits but is untested. With one gradient step per batch, ℓ = 0 when the gradient is taken, so CORO's β·ℓ term has zero gradient and PPO's ratio is always 1. CORO with β = 0.001 under staleness is unreliable in the same way (below).

## Mechanism (`scripts/mechanism.sh`)

Where does E5's speed come from? Three controls on seeds 0–2, with E4/E5's staleness and drift budget.

| config | change | final goal (seeds 0, 1, 2) | updates to 0.9 (per seed) | M turns to 0.9 |
|---|---|---|---|---|
| E5 Branched CORO | – | 1.00, 1.00, 1.00 | 351, 301, 301 | 1.31 |
| M1 | E5 without level 1: siblings only, prefix turns get no gradient | 1.00, 1.00, 1.00 | 301, 301, 251 | 1.17 |
| M2 | E5, but siblings redraw the goal change still pending at the fork | 1.00, 1.00, 1.00 | 351, 351, 351 | 1.47 |
| C1 | flat CORO on the same forked data | 0.91, 1.00, 1.00 | 651, 401, 401 | 2.00 |
| M3 | E4, but the 8 episodes of a group share one goal-change schedule, as siblings do | 0.17, 0.79, 0.98 | –, –, 651 | 2.67 (1 of 3) |

- Level 2 does the work. Without level 1, Branched CORO is at least as fast on every seed (284 updates on average against 318) with the same mean eval goal rate over the run (0.78). The fork turn is drawn per batch, so every turn position is a suffix in some batches and level 2 alone reaches every turn.
- Sharing the user's future is worth at most one eval interval. When siblings redraw the pending goal change, all three seeds reach 0.9 at 351, against 351, 301 and 301 for E5, and still ahead of every C1 seed.
- Shared randomness without a shared history does not help a flat loss. M3 reached 0.5 at updates –, 401 and 401 (E4: 501, 401, 351). Its drift ran slightly lower in the first 200 updates (mean KL 0.0012–0.0013 against 0.0014), so the dual lowered β sooner. On two seeds β hit the floor mid-learning (updates 391 and 604) and the goal rate stayed where it was (0.14 and 0.79). That is the failure from E7 and the β sweep.
- So the speed comes from contrasting continuations of the same state. In this env, level 1 and its soft-value gap (Limits) do not matter.

## Follow-up (`scripts/followup.sh`)

β fixed from the first update, no drift budget, staleness ≤ 11, seeds 0–2. Flat CORO at 0.3 is E3.

| β | flat CORO final | updates to 0.9 | Branched CORO final | updates to 0.9 |
|---|---|---|---|---|
| 0.001 | 0.74, 0.22, 0.35 | –, –, – | 0.82, 0.17, 0.83 | –, –, – |
| 0.03 | 0.92, 0.57, 0.98 | 551, –, 401 | 0.99, 1.00, 1.00 | 601, 701, 301 |
| 0.3 | 0.91, 1.00, 0.99 | 751, 551, 501 | 1.00, 1.00, 1.00 | 351, 351, 301 |
| 3 | 0.96, 0.75, 0.91 | 901, –, 951 | 0.99, 0.99, 0.98 | 651, 601, 701 |

- β matters. Adam undoes its overall scale, but β still sets how strongly ℓ is pulled toward the sampler compared with the reward. At 0.001 that pull is gone and CORO's gradient becomes group-centred policy gradient on stale data: no run reached 0.9. At 3 the pull is strong and learning is slow.
- 0.3, chosen earlier on seed 100, is the only value where all six runs reached 0.9, and the fastest for Branched CORO. Flat CORO at 0.03 was faster on the two seeds where it reached 0.9 (551 and 401 against 751 and 501), and missed on the third.
- Branched CORO is less sensitive to β. It reached 0.9 in all 9 runs with β from 0.03 to 3; flat CORO did in 4 of 6.
- So in CORO with a drift budget, β does two jobs: it is the dual variable the budget adjusts, and it is the anchor. Whenever drift is low, the dual lowers β and weakens the anchor. In E4 and E5, β reached the floor only after each seed had reached 0.9 (updates 556–969). In E7 it got there at updates 402–431, before any seed reached 0.9. The KKT conditions predict this: a multiplier on a slack constraint goes to zero, and drift was small here, so anything else that holds drift down takes the anchor away. Next time I would keep β as a fixed anchor and give the budget its own multiplier.

### 32-turn episodes

E4, C1 and E5 with everything else as in the main matrix (staleness up to 15, since episodes now span more sampler updates). 1000 updates, about 8.2M agent turns per run. β and the budget were not retuned.

| config | final goal (seeds 0, 1, 2) | updates to 0.5 | updates to 0.9 |
|---|---|---|---|
| E4 CORO + budget | 0.89, 0.19, 0.72 | 751, –, 901 | –, –, – |
| C1 flat CORO on forked data | 0.78, 0.66, 0.37 | 601, 851, 1000 | –, –, – |
| E5 Branched CORO + budget | 0.78, 0.83, 0.98 | 401, 651, 601 | –, –, 851 |

- All three learn at 32 turns, which the earlier synchronous runs never did, but none finished learning in 1000 updates. Only one run reached 0.9.
- The ranking is the same as at 16 turns. Branched CORO averages 0.86 against 0.60 for both others and reaches 0.5 first on every seed. The final scores overlap across seeds, so this does not show whether the gap grows with horizon.

## Limits

- One synthetic env with full context visible. The main results are at 16 turns; at 32 turns the runs had not finished learning after 1000 updates, and nothing longer was run with staleness. With context shorter than the episode, no method learned. Writing and recalling pay nothing until both work, so that is an exploration problem more than a credit one.
- A 64-wide, 2-layer policy and 32 agent tokens per episode. This is far from the token counts where IS ratios are reported to explode.
- 3 seeds per configuration. Failure analysis uses the 8 saved held-out episodes per run (`trajectories.txt`).
- `fork` copies the env state by default, including a goal change scheduled for later, so siblings face the same user after the fork. With `--fork_resample` they redraw it (M2), which cost at most one eval interval. Users whose every reply is random are not tested.
- The β sweep uses Adam with gradient clipping at staleness ≤ 11. The best β will move with drift size and reward scale.
- Level 1 of Branched CORO uses the plain mean of the continuation returns. At CORO's fixed point the prefix target is the soft value r_pre + β·log mean exp(r_suf/β): with β = 0.3 and 2 of 4 continuations succeeding, 0.80 instead of 0.50. With k = 4 that estimate is biased too. The switch is untested; M1 shows level 1 adds nothing measurable here, so the gap does not affect the results.

## History

The scripts record how the setup was reached. Each one's header says what it tested.

- `day1.sh`: first learnability check. At 64 turns with truncated context, the policy collapsed to one answer.
- `day2*.sh`: env v2, the first β sweep on seed 100 (0.03 → 0.89, 0.1 → 0.43, 0.3 → 1.00, 1.0 → 0.13), and the switch to full context. That sweep ran synchronously, where ℓ = 0 when the gradient is taken, so it could hardly tell β values apart, and it used one seed per value. In the later fixed-β sweep on seeds 0–2, 0.3 is the only value where all six runs reached 0.9. The first env v2 runs collapsed to always-PASS. Only the ANSWER verb was rewarded, and std-normalised advantages inflated the 0.01 speech cost. The final reply is now scored whatever its verb.
- `day2_fullctx.sh`: synchronous runs with full context at 32 and 64 turns, seed 100. Nothing learned (goal rate 0.12–0.25).
- `main.sh`, `severity.sh`, `e7.sh`, `controls.sh`, `followup.sh`, `mechanism.sh`: the reported experiments.

The run folders under `runs/` from these stages are kept, including the failed ones.
