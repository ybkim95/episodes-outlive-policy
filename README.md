# Episodes that outlive the policy

Two components of an async RL stack for long, multi-turn agent episodes, on a small testbed:

1. **Trajectory record** (`traj.py`, `runner.py`): every turn stores the policy version that sampled it, the exact context it saw, behaviour log-probs, and the fork point when an episode is branched. The env copies its own state at that turn, so the k continuations start from one mid-episode state.
2. **Off-policy objectives** (`offpolicy.py`): PPO-clip, OAPL, CORO with a drift budget, and **Branched CORO** (`tree_loss`), which forks an episode at turn *b* into *k* continuations and contrasts suffixes among siblings. Per-turn drift accounting and three ways of enforcing a per-turn budget: the first two hurt; the third, with β held fixed, works (see below).

Staleness is controlled and differs *within* an episode. Sampler weights advance every `U` turns, and each episode also gets a heavy-tailed queue delay `q ≤ qmax`, so early turns are older than late ones.

In `runner.py` the sampler and trainer are not separate processes. Past trainer weights sit in a ring buffer, and each turn is sampled with the weights its staleness points to. Staleness is therefore exact, and every run repeats bit for bit. `async_runner.py` runs the same record and losses in a real asynchronous loop: sampler processes, a bounded queue, and weight swaps at turn boundaries. There nobody sets the staleness; it is measured per turn (results below).

## In a real async stack

What the record has to handle with a queue between sampler and trainer workers. `async_runner.py` implements the first two rows; the last two are designed, not built.

| issue | what goes wrong | design |
|---|---|---|
| weights change mid-episode | the sampler's KV cache was built by older weights | swap at turn boundaries; re-prefill or keep the old cache, but always store the log-probs the sampler computed, which are μ. Any numeric mismatch between the inference engine and the trainer then shows up in ℓ, where the loss sees it, so fresh turns are never exactly on-policy |
| long episodes are staler | a cutoff on episode age drops long episodes first | bound the age of the last turn; per-turn versions keep the older turns usable |
| branch groups wait on stragglers | a forked group is usable only when every continuation has finished | close each group at a deadline: any two continuations still give a contrast |
| context storage | storing each turn's full context grows with the square of the turns | per-turn deltas plus a prefix pointer; forks share the prefix and its KV cache |

Both runners re-score every turn with that turn's weights, which is the re-prefill option. `Rollout` currently stores full per-turn contexts. In `async_runner.py` the sampler computes μ on CPU and the trainer scores on GPU, so fresh turns carry a numeric log-ratio, but a tiny one here: per-token KL at most 3e-9, against about 1e-3 for stale turns. `runner.py --sampler_bits 4` makes that mismatch large on purpose (below).

## Layout

| file | what it is |
|---|---|
| `env.py` | `MemoryDialogue`: user, tool, teammate, memory slot, mid-episode goal change, reward only on the final reply, 0.01 cost for speaking early |
| `traj.py` | `Rollout` record and `build_context` (exact per-turn context) |
| `policy.py` | tiny causal transformer (d = 64, 2 layers); `act` and `score` over agent tokens only |
| `credit.py` | outcome advantages (GRPO-style and Dr.GRPO-style) for the `none`/`ppo` objectives |
| `offpolicy.py` | `loss` (none / ppo / oapl / coro), `tree_loss` (Branched CORO), `tree_advantages` (branched PPO-clip baseline), `turn_kl_penalty`, `drift_stats` |
| `async_runner.py` | real asynchronous loop: N sampler processes, a bounded queue of whole batches, shared weights with a version counter, swaps at turn boundaries, measured per-turn staleness, `max_age` |
| `runner.py` | per-turn staleness, `--sampler_bits` (sampler on rounded weights), versioned rollout with forking, two-pass chunked backward, training loop, dual ascent, the control modes (`tree_flat`, `tree_ppo`, `--beta_path`, `--tree_no_level1`, `--fork_resample`, `--crn`, `--minibatches`) |
| `tests/` | 34 tests (`test_sanity.py`, `test_async.py`) that must pass before any number is trusted |
| `analyze.py` | every table in the write-up (main, severity, E7, full-run drift, controls, follow-ups), from `runs/*/metrics.jsonl` only |
| `talk_figs.py` | slide-sized figures (`figs/talk_*.png`) |
| `plots.py` | ad-hoc curves for any run glob |
| `scripts/` | the exact launch scripts (see History) |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests/                 # 34 tests, ~20 s on CPU
bash scripts/main.sh a                     # E0-E3 on seeds 0-2, drift-budget tuning on seed 100
bash scripts/main.sh b 0.002 1             # E4-E6 with the chosen budget
bash scripts/severity.sh                   # staleness up to 31 and 63
bash scripts/e7.sh                         # per-turn KL penalty (E7), three step sizes
bash scripts/controls.sh                   # controls C1-C5 and reruns of E4/E5 (needs the E5 and E7 runs)
bash scripts/followup.sh                   # fixed-beta sweep, and E4/C1/E5 at 32 turns (about 1 h at 32 turns)
bash scripts/mechanism.sh                  # M1-M3, first (unpaired) version
bash scripts/review.sh                     # paired M1-M3, branched PPO-clip, sync with 4 steps per batch
bash scripts/fix.sh                        # X1 (per-turn penalty with beta fixed), X2 (Branched CORO, no staleness)
bash scripts/async.sh                      # real async sampler and trainer, three methods
bash scripts/review2.sh                    # T256, E4/E5/C1/C2/T256 on 8 seeds, PPO-clip tuning, X1g
bash scripts/lowprec.sh                    # sampler on 4-bit weights
bash scripts/ppo2.sh                       # tuned PPO-clip on seeds 0-2 (after review2.sh)
python analyze.py                          # tables + figs/main_*.png, severity.png, e7_drift.png, controls.png, beta_sweep.png
python talk_figs.py                        # figs/talk_*.png
```

The scripts assume three GPUs, numbered 1–3; edit `GPUS=(...)` at the top of a script to change that. One run takes about 25–45 minutes on a shared GPU; the model is tiny, so CPU works too (`--device cpu`), just slower.

Protocol: β, the learning rate (1e-3 over 3e-3) and the drift budget were chosen on seed 100 only, and every reported number is from seeds 0–2 (0–7 for the 8-run reruns). The clip range (0.2), OAPL's β₁ (1), and the branch sizes (k = 4 continuations, 4 trunks per seed) are defaults I never tuned. The drift budget is fragile: of the four (budget, dual step) settings tried on seed 100, only δ = 0.002 with step 1 learned (0.83); the other three ended at chance (0.13). It also adds little to Branched CORO. At fixed β = 0.3 with no budget, Branched CORO reaches 0.9 at 351, 351, 301 updates against 351, 301, 301 for E5. For flat CORO it helps more: 751, 551, 501 without (E3) against 651, 501, 401 with (E4). Runs are matched on agent turns generated (≈4.1M each, branched runs included), except C2, which uses 1.6× the turns per update by design.

Trainer cost is not matched. A branched batch has 416 episodes against 256, and the trainer scores every turn of every episode, so it scores 1.6× the contexts per update. On seeds 0–2 that left E4 and E5 about even in trainer compute. Over 8 runs E5 needs half the updates (median 311 against 611), so it scores about a sixth fewer episodes in all (129k against 156k). The larger saving is in sampled turns, which the prompt names as the expensive part. C1 and C2 have E5's trainer cost per update. Shared prefixes are scored k times. Scoring them once would bring a branched update to the same number of scored turns as E4 (about 4.2k against 4.1k), but about 1.2× its context tokens, because the shared turns are the short early ones. At the 8-run medians, E5 would then need about 40% less trainer compute than E4. Not implemented.

Provenance: each run's `config.json` has `src_hash`. Until the controls it hashed every `.py` file, analysis scripts included, so it changed between stages without any training change. It now hashes only the six training files. E4 and E5 rerun on later code match the original runs' loss bit for bit for all 1000 updates, seeds 0–2. The control flags added after that leave the default path unchanged, which I checked the same way.

## Results (H = 16, full context, staleness 0–11 within episodes)

Run codes: E0–E7 are the main matrix, C1–C5 the controls, M1p–M3p the mechanism controls, B1 the branched PPO-clip baseline, S1–S3 the synchronous runs with several steps per batch. In the talk, E4 is "CORO + drift budget", E5 is "Branched CORO + budget", E6 and E7 are the two per-turn budgets, C1 is "same forked data, flat CORO", C2 is "416 independent episodes", and M1p is "level 2 only".

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

- Branched CORO reaches 0.9 about 1.6× sooner than CORO with the same drift budget on these three seeds (2.0× by the median of 8 runs each, below: 1.27M against 2.50M turns). Two things contribute about equally: fewer updates, and fewer sampled turns per update because forks share their prefix. At the same batch size (T256, below) Branched CORO still needs fewer updates, but one run in 8 stalled. The controls below trace the update savings to the sibling contrast (level 2): on the same forked data a flat loss needs more updates (1.24× by the median of 8 runs, slower than E5 on every seed), level 2 alone is within one eval interval of the full loss, and PPO-clip on the same branches is slower untuned and close once tuned (below).
- Up to staleness 63 (`severity.sh`), ignoring staleness falls to chance, while PPO-clip, OAPL and CORO all average 0.93 or more at every level. PPO-clip did not become unstable in this testbed at full precision; with a 4-bit sampler at staleness ≤ 31 it slips to 0.87 (below). Masking was not tested. At staleness ≤ 63, all three seeds that ignore staleness collapsed to one fixed action per turn (for example `PASS G6`), so every ratio is 1, their trajectory ESS reads 1.00, and the final reply is one fixed goal. Not all of that failure is staleness: plain policy gradient is weak here even without it (C5, synchronous: 0.53, 0.34, 0.86; on seed 100 at learning rates 1e-3 and 3e-3 it ends near 0.35). Staleness makes it worse (0.23).
- Drift stayed small, with rare spikes. Over every update of every run, the median trajectory ESS is at least 0.77 of the batch for every corrected method, and the 5th percentile at least 0.59. The largest token ratio per update has a median of at most 1.55, but its maximum reaches 15–214 depending on method and staleness (full-run drift table in `python analyze.py`).
- The first two ways of enforcing a per-turn budget hurt, and both mostly fail on goal-change episodes. The third (X1, below) works.
  - E6 gives each staleness bucket its own β. A turn's β also scales its gradient. Fresh turns have no drift, so their β fell to the floor, and the rewarded final turn (in a fresh bucket 79% of the time) learned far more slowly: the mean goal rate was still climbing at 0.79 after 1000 updates.
  - E7 keeps one β and adds a KL penalty per bucket. The penalty kept KL under budget, so the dual pushed β to its floor at updates 402–431 while the run was still learning. Training then nearly stopped. CORO's gradient is proportional to β (the gradient of (β·Δℓ − Δr)² is 2β(β·Δℓ − Δr)∇Δℓ), so the gradient norm fell from 3–6 to about 0.016. Adam's second-moment average (decay 0.999, a memory of about 1000 updates) still held the earlier gradients, so its steps shrank too, and per-token KL fell from about 1e-3 to 5e-6. The controls and the β sweep below confirm it.

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

- On the same forked data, flat CORO (C1) needs 484 updates on average and Branched CORO (E5) needs 318. More independent episodes per update (C2) help per update (451) but cost 1.6× the turns. On these seeds every E5 run is faster than every run of E4, C1 and C2. Over 8 runs (next section) the C1 gap shrinks to 1.24× by the median, still with E5 faster on every paired seed (sign test p = 0.008), and C2 needs 2.1× the turns (p < 0.001). So the loss accounts for the savings in updates, and the mechanism controls below narrow that down to level 2. The batch-size question is settled by T256 (next section): it is not the extra endings. The branched PPO-clip baseline (B1, below) shows how much of the gain is CORO's.
- Replaying E5's β path into E7 (C3) brings it to 0.96–0.99, though seed 0 is still slow. E7's own β reached the 1e-3 floor at updates 403–432. E5's stayed between 0.18 and 0.34 until E5 reached 0.9.
- C4 left E5 unchanged. I first read that as "β is inert under Adam", and that was wrong. Before E5 reached 0.9, the two β paths differed by at most 2.8× on seeds 1 and 2 and 5.4× on seed 0, so C4 only shows that E5 tolerates that much. The fixed-β sweep below is the real test.
- Both synchronous runs are slow: sync policy gradient (C5) matches sync CORO (E0). With one gradient step per batch, ℓ = 0 when the gradient is taken, so CORO's β·ℓ term has zero gradient and PPO's ratio is always 1. S1–S3 below test this. CORO with β = 0.001 under staleness is unreliable in the same way (below).

## Same batch size, and 8 runs (`scripts/review2.sh`)

Two doubts remain after the controls. A branched batch has 416 rewarded episodes per update against E4's 256, and three runs per side cannot rule out luck. T256 is Branched CORO with 256 episodes per update (16 simulation seeds × 4 trunks × 4 continuations), the same as E4. E4, E5, T256 and the controls C1 and C2 were rerun on 8 training seeds with an eval every 10 updates. Seeds 0–2 of E4 and E5 reproduce the earlier runs' training loss bit for bit; the eval draws from its own random stream.

| config | episodes per update | seeds done | reached 0.9 | updates to 0.9 (per seed) | median | k rewarded episodes to 0.9 (median) | M turns to 0.9 (median) | rank test vs E4: updates, turns | rank test vs E5: updates, turns |
|---|---|---|---|---|---|---|---|---|---|
| E4 CORO + budget | 256 | 8 | 7 | 651, 461, 401, 741, 621, -, 601, 451 | 611 | 156 | 2.50 |  |  |
| T256 Branched CORO + budget | 256 | 8 | 7 | -, 361, 361, 471, 381, 791, 311, 321 | 371 | 95 | 0.95 | p = 0.137, 0.021 |  |
| E5 Branched CORO + budget | 416 | 8 | 8 | 331, 291, 281, 361, 331, 441, 261, 261 | 311 | 129 | 1.27 | p = 0.000, 0.001 |  |
| C1 flat CORO on E5's forked data | 416 | 8 | 7 | 571, 391, 381, 501, 341, -, 351, 321 | 386 | 161 | 1.59 | p = 0.052, 0.052 | p = 0.019, 0.021 |
| C2 CORO + budget, independent episodes | 416 | 8 | 8 | 391, 501, 441, 421, 361, 781, 381, 341 | 406 | 169 | 2.70 | p = 0.038, 0.382 | p = 0.006, 0.000 |

The rank test is exact: a two-sided permutation test on mean ranks over every split of the pooled runs, with a run that never reached 0.9 ranked slowest. C1 runs on E5's data stream seed by seed, so it also gets a paired sign test.

- At the same batch size, Branched CORO needs fewer updates: median 371 against 611. That is about 1.6× fewer rewarded episodes (95k against 156k), so the extra endings in a 416-episode batch do not explain the gap.
- It also samples fewer turns per update (2,560 against 4,096), because the 64 trunks share their prefixes. In sampled turns it reaches 0.9 with a median of 0.95M against 2.50M (p = 0.02).
- It is less reliable at this batch size. One T256 run stalled at 0.79 for 500 updates and ended at 0.85, and another took 791 updates, so in updates alone the gap to E4 is not significant (p = 0.14). The stalled run answers the goal from before the user's change in some goal-change episodes. At 416 episodes every E5 run reached 0.9, by update 441 at the latest (p < 0.001 against E4, in updates and in turns).
- The controls hold at 8 runs, with smaller gaps than on seeds 0–2. On the same forked data, flat CORO (C1) needs a median of 386 updates against E5's 311 (1.24×, not the 1.5× of the first three seeds), and E5 is faster on all 8 paired seeds (sign test p = 0.008; rank test p = 0.02). One C1 run never reached 0.9. Independent episodes at the same batch size (C2) need 406 updates and 2.1× E5's sampled turns (2.70M against 1.27M, p < 0.001).
- More runs make E4 look slower than its first three seeds did: median 611 updates and 2.50M turns, against a mean of 518 and 2.12M on seeds 0–2. One E4 run of 8 never reached 0.9.

## Mechanism (`scripts/review.sh`)

Where does E5's speed come from? Each control runs on the same random stream as its reference (same goals, staleness and fork turns, update by update; tested) and replays the reference's β path, so neither the data nor the dual can differ by accident. Seeds 0–2.

| config | change | final goal (seeds 0, 1, 2) | updates to 0.9 (per seed) | M turns to 0.9 |
|---|---|---|---|---|
| E5 Branched CORO | – | 1.00, 1.00, 1.00 | 351, 301, 301 | 1.31 |
| M1p | E5 without level 1: siblings only, prefix turns get no gradient | 1.00, 1.00, 1.00 | 351, 351, 351 | 1.44 |
| M2p | E5, but siblings redraw the goal change still pending at the fork | 1.00, 0.99, 1.00 | 351, 351, 401 | 1.51 |
| C1 | flat CORO on the same forked data | 0.91, 1.00, 1.00 | 651, 401, 401 | 2.00 |
| E4 | flat CORO, 256 independent episodes | 0.91, 0.99, 0.99 | 651, 501, 401 | 2.12 |
| M3p | E4, but the 8 episodes of a group share one goal-change schedule, as siblings do | 1.00, 0.90, 0.89 | 551, 551, – | 2.26 (2 of 3) |

- Level 2 does most of the work. Without level 1, Branched CORO needs 351 updates on every seed against 351, 301 and 301, so level 1 is worth at most one eval interval. The fork turn is drawn per batch from turns 1–15, so every turn except the first is a suffix in some batches and level 2 alone reaches it. Turn 0 gets no gradient in M1p, and in 6 of its 24 saved episodes it opens with something other than LOOKUP (E5: 1 of 24).
- The shared user future is a small part. When siblings redraw the pending goal change, 0.9 takes 368 updates on average against 318, and C1 needs 484.
- A shared user future without a shared history does not help a flat loss: M3p reaches 0.9 at 551, 551 and never, against 651, 501 and 401 for E4.
- So most of the gain comes from contrasting continuations of the same state. In this env, level 1 and its soft-value gap (Limits) barely matter.
- The first version of these controls (`mechanism.sh`, runs `k_*`) was not paired: `--crn` and `--fork_resample` consumed a different number of random numbers, and β followed each run's own dual. It gave 284 updates for level 2 only, 351 for the redraw, and 1 of 3 seeds for the shared schedule, where β hit its floor on the other two. The paired runs above replace it. Rerunning `mechanism.sh` on the current code gives paired runs without the β replay; the `k_*` runs come from commit 55a2088.

### Is it branching or CORO? (B1)

Same forked data as E5, PPO-clip (clip 0.2) on the advantages the two levels imply: suffix return standardised among siblings, prefix value standardised across trunks (`tree_advantages`). No drift budget, so the reference is Branched CORO at fixed β = 0.3 with no budget, on the same random stream.

| config | final goal (seeds 0, 1, 2) | updates to 0.9 (per seed) | M turns to 0.9 |
|---|---|---|---|
| E1 flat PPO-clip | 0.99, 1.00, 0.98 | 651, 551, 601 | 2.46 |
| B1 branched PPO-clip | 0.88, 1.00, 1.00 | –, 451, 401 | 1.76 (2 of 3) |
| Branched CORO, no budget | 1.00, 1.00, 1.00 | 351, 351, 301 | 1.38 |

- Branching helps PPO-clip too, on the two seeds that reached 0.9 (451 and 401 against 551 and 601). Seed 0 stalled at 0.88.
- On the same branches, CORO is faster on every seed than this untuned baseline. Tuned (next section), branched PPO-clip comes close: 351, 451, 351 against 351, 351, 301. So the branching does most of the work. CORO is never slower on a seed, but by one or two eval intervals at most.

### PPO-clip tuned as CORO was (`scripts/review2.sh`, `scripts/ppo2.sh`)

E1 and B1 used standardised advantages and clip 0.2 untuned. Stage 1 tried standardised (A0) and only centred (A0dr, the same scale as CORO's targets) advantages with clip 0.1, 0.2 and 0.3 on seed 100; final goal and updates to 0.9:

| advantages | clip | flat PPO-clip: final, updates to 0.9 | branched PPO-clip: final, updates to 0.9 |
|---|---|---|---|
| standardised | 0.1 | 1.00, 601 | 1.00, 451 |
| standardised | 0.2 | 0.93, 651 | 1.00, 401 |
| standardised | 0.3 | 1.00, 551 | 1.00, 801 |
| centred only | 0.1 | 1.00, 701 | 1.00, 351 |
| centred only | 0.2 | 0.99, 501 | 1.00, 301 |
| centred only | 0.3 | 0.80, never | 1.00, 301 |

Centred-only advantages with clip 0.2 were fastest for both (flat 501, branched 301; for branched, clip 0.3 tied and the tie kept the default clip). Stage 2 ran that setting on seeds 0–2, no drift budget, staleness ≤ 11:

| config (no drift budget, staleness <= 11) | final goal (seeds 0, 1, 2) | updates to 0.9 (per seed) |
|---|---|---|
| E1 flat PPO-clip, default (standardised, clip 0.2) | 0.99, 1.00, 0.98 | 651, 551, 601 |
| flat PPO-clip, tuned (centred only, clip 0.2) | 0.79, 1.00, 0.98 | -, 501, 401 |
| E3 flat CORO | 0.91, 1.00, 0.99 | 751, 551, 501 |
| B1 branched PPO-clip, default | 0.88, 1.00, 1.00 | -, 451, 401 |
| branched PPO-clip, tuned (centred only, clip 0.2) | 0.97, 1.00, 1.00 | 351, 451, 351 |
| Branched CORO, beta 0.3 | 1.00, 1.00, 1.00 | 351, 351, 301 |

- Centring only, like CORO's targets, is faster for both. A likely reason, not tested: standardising divides by each group's reward spread, so in a group where every episode fails, the 0.01 speech cost becomes a full-size advantage (the env v2 collapse in History came from the same effect).
- Tuned branched PPO-clip reaches 0.9 on all three seeds (351, 451, 351), against 351, 351, 301 for Branched CORO on the same stream. Branching does most of the work, and CORO's regression adds at most two eval intervals here.
- Tuned flat PPO-clip is faster where it learns (501 and 401) but one run of three stalled at 0.79, so E1 stays the flat reference. No flat comparison in this write-up changes.

### Synchronous training, several steps per batch (S1–S3)

Why do the synchronous runs (E0, C5) lose to stale ones? Same as E0/C5, but each batch is split into 4 minibatches of whole groups, one gradient step each. Trainer compute per sampled turn stays the same. From the second step on, the stored log-probs are behind the trainer, so ℓ ≠ 0 and CORO's anchor and PPO's clip can act. Plain policy gradient has neither, so it is the control.

| config | final goal (seeds 0, 1, 2) | batches to 0.9 (per seed) | M turns to 0.9 |
|---|---|---|---|
| E0 sync CORO, 1 step per batch | 0.52, 0.37, 1.00 | –, –, 451 | 1.85 (1 of 3) |
| S1 sync CORO, 4 steps per batch | 1.00, 1.00, 1.00 | 401, 351, 751 | 2.05 |
| C5 sync policy gradient, 1 step | 0.53, 0.34, 0.86 | –, –, – | – |
| S3 sync policy gradient, 4 steps | 0.22, 0.26, 0.12 | –, –, – | – |
| S2 sync PPO-clip, 4 steps | 0.75, 0.48, 0.57 | –, –, – | – |

- With 4 steps per batch, sync CORO reaches 1.00 on every seed, as fast as the stale CORO runs (E3: 2.46M turns, E4: 2.12M). The same split makes policy gradient worse (0.20 on average against 0.58). So the extra steps are not what helps; the anchor is. With one step per batch ℓ = 0, CORO's anchor does nothing, and sync CORO is plain policy gradient.
- PPO-clip does not recover (0.60 on average). Stale PPO-clip (E1) reaches 0.99, so for PPO the gap between sync and stale is still unexplained.

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
- Branched CORO reached 0.9 in all 9 runs with β from 0.03 to 3; flat CORO did in 7 of 9. That is too few runs to call it less sensitive to β.
- So in CORO with a drift budget, β does two jobs: it is the dual variable the budget adjusts, and it is the anchor. Whenever drift is low, the dual lowers β and weakens the anchor. In E4 and E5, β reached the floor only after each seed had reached 0.9 (updates 557–970). In E7 it got there at updates 403–432, before any seed reached 0.9. The KKT conditions predict this: a multiplier on a slack constraint goes to zero, and drift was small here, so anything else that holds drift down takes the anchor away. X1 (below) does that: β fixed at 0.3, the budget enforced by the per-bucket multipliers alone. It reaches 1.00 on every seed.

### 32-turn episodes

E4, C1 and E5 with everything else as in the main matrix (staleness up to 15, since episodes now span more sampler updates). 1000 updates, about 8.2M agent turns per run. β and the budget were not retuned.

| config | final goal (seeds 0, 1, 2) | updates to 0.5 | updates to 0.9 |
|---|---|---|---|
| E4 CORO + budget | 0.89, 0.19, 0.72 | 751, –, 901 | –, –, – |
| C1 flat CORO on forked data | 0.78, 0.66, 0.37 | 601, 851, 1000 | –, –, – |
| E5 Branched CORO + budget | 0.78, 0.83, 0.98 | 401, 651, 601 | –, –, 851 |

- All three learn at 32 turns. The synchronous 32-turn runs in `day2_fullctx.sh` (seed 100, a different setup) never did, so this is not a controlled comparison. None finished learning in 1000 updates. Only one run reached 0.9.
- The ranking is the same as at 16 turns. Branched CORO averages 0.86 against 0.60 for both others and reaches 0.5 first on every seed. The final scores overlap across seeds, so this does not show whether the gap grows with horizon.

## Fixing the per-turn budget, and branching without staleness (`scripts/fix.sh`)

X1 is E7 with β held at 0.3 (`--beta_lr 0`), so only the per-bucket KL multipliers move. Its step was chosen on seed 100 from 100, 1000 and 10000, as E7's was. X2 is Branched CORO with no staleness (`--sync`): does branching help only on stale data?

| X1 step | seed 100 (tuning) | seeds 0, 1, 2 | updates to 0.9 (seeds 0–2) |
|---|---|---|---|
| 100 (chosen) | 1.00 | 1.00, 1.00, 1.00 | 451, 351, 401 |
| 1000 | 1.00 | 1.00, 1.00, 1.00 | 651, 501, 451 |
| 10000 | 0.84 | 0.88, 0.84, 0.99 | –, 751, 601 |

| config | final goal (seeds 0, 1, 2) | updates to 0.9 (per seed) |
|---|---|---|
| E7 per-turn penalty, β by dual ascent | 0.75, 0.78, 0.93 | –, –, 801 |
| X1 per-turn penalty, β fixed at 0.3 | 1.00, 1.00, 1.00 | 451, 351, 401 |
| X1g one global KL multiplier, β fixed at 0.3 (step 100, not retuned) | 1.00, 1.00, 1.00 | 451, 351, 301 |
| Branched CORO, β 0.3, no budget | 1.00, 1.00, 1.00 | 351, 351, 301 |
| E0 CORO, no staleness | 0.52, 0.37, 1.00 | –, –, 451 |
| X2 Branched CORO, no staleness | 1.00, 1.00, 0.97 | 751, 401, 651 |

- With β fixed, the per-turn budget works. X1 reaches 1.00 on every seed. Median per-token KL in each staleness bucket stays at 0.6–1.6e-3, under the 2e-3 budget, and 2–6% of updates exceed the budget, against 6–11% with no budget. The price is about one eval interval against the same runs with no budget. One global multiplier instead of one per staleness level (X1g, same step) works as well, so the fix is holding β fixed, not the per-turn split.
- Branching does not need stale data. Without staleness, ℓ = 0 when the gradient is taken, so Branched CORO becomes policy gradient with sibling and trunk baselines. It still reaches 0.97–1.00 on every seed, where flat sync CORO (E0) reaches 0.9 on one seed of three. It is slower than with staleness (601 against 334 updates on average): as for flat CORO (S1), the anchor does nothing at one gradient step per batch.

## 4-bit sampler (`scripts/lowprec.sh`)

The regime CORO was built for: the sampler disagrees with the trainer even on fresh turns. The sampler runs weights rounded to 4 bits (blocks of 16, absmax scale; `--sampler_bits 4`) and the trainer full precision, as with low-precision rollouts. At initialisation this alone raises per-token KL from about 1e-3 to 8e-3. No drift budget; β is fixed at 0.3. The full-precision references are E1, E3, E1b, Branched CORO at fixed β = 0.3, and the staleness ≤ 31 severity runs. There is no full-precision Branched CORO at staleness ≤ 31. Final goal rate, seeds 0–2:

| objective | staleness <= 11: full precision | 4-bit sampler | staleness <= 31: full precision | 4-bit sampler |
|---|---|---|---|---|
| PPO-clip | 0.99, 1.00, 0.98 | 0.99, 0.97, 0.99 | 0.97, 0.99, 1.00 | 0.79, 0.87, 0.95 |
| CORO | 0.91, 1.00, 0.99 | 0.93, 0.12, 0.93 | 0.99, 0.99, 0.99 | 0.85, 0.80, 0.99 |
| Branched CORO | 1.00, 1.00, 1.00 | 0.99, 1.00, 0.78 | - | 0.99, 0.99, 0.94 |
| ignore staleness | 0.24, 0.20, 0.25 | 0.47, 0.14, 0.35 | 0.23, 0.14, 0.36 | 0.17, 0.20, 0.22 |

- At staleness ≤ 11 the rounding costs PPO-clip nothing. CORO and Branched CORO each lose one run. That CORO run collapsed to always answering G6 (0.12) while its KL and ESS stayed normal, so it is the fixed-answer collapse seen when staleness is ignored, not an importance-ratio blow-up. The Branched CORO run stalled at 0.78.
- At staleness ≤ 31 every method slips. PPO-clip falls from 0.99 to 0.87 on average and reaches 0.9 on two runs (701 and 1000 updates); CORO falls to 0.88 and reaches it on one (751). Branched CORO ends at 0.97 and reaches 0.9 on all three (401, 501, 351).
- There is still no blow-up. Per-token KL roughly doubles (PPO-clip at staleness ≤ 31: 3.4e-3 to 5.5e-3), and the median trajectory ESS falls to 0.73–0.77 for the corrected methods and 0.30 when staleness is ignored. The damage shows up as slower learning and stalls. Three runs per cell.

## Real async sampler and trainer (`scripts/async.sh`, `scripts/async2.sh`)

`async_runner.py` runs the same record and losses in separate processes: 8 sampler processes on CPU, a queue of at most 2 whole batches, and a trainer on GPU that publishes new weights after every update. Samplers pull weights only at turn boundaries and record the version per turn. A 50 ms pause per turn stands in for the user model and tools. Batches whose last turn is more than 32 updates old would be dropped; none were. Settings are E4's, E5's and E1's, nothing re-tuned. Seeds 0–2. The machine was shared and its load changed over the day, so wall-clock minutes are not comparable across runs; a_ppo_s0 ran alone after the others (`async2.sh`); its staleness matched the other runs'.

| async | final goal (per seed) | updates to 0.9 | M turns to 0.9 | minutes to 0.9 | staleness: first turn, last turn, max | trainer idle |
|---|---|---|---|---|---|---|
| CORO + drift budget | 0.91, 0.99, 0.99 | 651, 601, 551 | 2.67, 2.46, 2.26 | 19, 28, 21 | 8.4, 1.4, 13 | 51% |
| Branched CORO + budget | 0.80, 1.00, 1.00 | -, 401, 401 | -, 1.65, 1.69 | -, 23, 19 | 8.4, 1.6, 16 | 59% |
| PPO-clip | 0.73, 0.99, 1.00 | -, 701, 701 | -, 2.87, 2.87 | -, 34, 25 | 8.5, 1.4, 13 | 53% |

- Staleness, measured: the first turn of an episode is 8.4 updates stale on average and the last 1.4–1.8; the worst turn reached 13–16, and an episode spans 6.4–7.5 trainer versions. That is the regime the ring buffer simulated (staleness ≤ 11), a little staler.
- The ranking holds, with stalls. Branched CORO reaches 0.9 at 401 updates on two runs, CORO + budget at 551–651 on all three, PPO-clip at 701 on two. The third Branched CORO run stalled at 0.80, and the third PPO-clip run at 0.73, both in the same way as the stalled T256 run: after the user changes the goal, they sometimes answer the old one. So Branched CORO's final mean (0.93) ends below CORO's (0.96). Mean staleness per turn is 5.6 here against 2.9 in the simulated runs, and the goal change is the last skill learned.
- The drift budget binds here, unlike in the simulated runs. Median per-token KL sits at the 2e-3 budget, and the dual raises β to 0.55–0.8 instead of lowering it (0.18–0.34 before 0.9 in E5). A larger β is a stronger anchor, which slows learning in the fixed-β sweep; that may be why Branched CORO needs 401 updates here against a median of 331 simulated.
- The trainer waited for data 51–59% of the time: 8 CPU samplers could not keep one trainer busy, as the prompt expects of rollouts.
- Fresh turns are scored by a different forward pass (sampler on CPU, trainer on GPU), and their per-token KL is at most 3e-9.

## Limits

- One synthetic env with full context visible. The main results are at 16 turns; at 32 turns the runs had not finished learning after 1000 updates, and nothing longer was run with staleness. With context shorter than the episode, no method learned. Writing and recalling pay nothing until both work, so that is an exploration problem more than a credit one.
- A 64-wide, 2-layer policy and 32 agent tokens per episode. This is far from the token counts where IS ratios are reported to explode.
- 3 seeds per configuration, 8 for E4, E5, T256, C1 and C2. Failure analysis uses the 8 saved held-out episodes per run (`trajectories.txt`).
- `fork` copies the env state by default, including a goal change scheduled for later, so siblings face the same user after the fork. With `--fork_resample` they redraw it (M2p), and 0.9 takes 368 updates on average instead of 318. Users whose every reply is random are not tested.
- The β sweep uses Adam with gradient clipping at staleness ≤ 11. The best β will move with drift size and reward scale.
- Level 1 of Branched CORO uses the plain mean of the continuation returns. At CORO's fixed point the prefix target is the soft value r_pre + β·log mean exp(r_suf/β): with β = 0.3 and 2 of 4 continuations succeeding, 0.80 instead of 0.50. With k = 4 that estimate is biased too. The switch is untested; M1p shows level 1 is worth at most one eval interval here, so the gap barely matters.

## History

The scripts record how the setup was reached. Each one's header says what it tested.

- `day1.sh`: first learnability check. At 64 turns with truncated context, the policy collapsed to one answer.
- `day2*.sh`: env v2, the first β sweep on seed 100 (0.03 → 0.89, 0.1 → 0.43, 0.3 → 1.00, 1.0 → 0.13), and the switch to full context. That sweep ran synchronously, where ℓ = 0 when the gradient is taken, so it could hardly tell β values apart, and it used one seed per value. In the later fixed-β sweep on seeds 0–2, 0.3 is the only value where all six runs reached 0.9. The first env v2 runs collapsed to always-PASS. Only the ANSWER verb was rewarded, and std-normalised advantages inflated the 0.01 speech cost. The final reply is now scored whatever its verb.
- `day2_fullctx.sh`: synchronous runs with full context at 32 and 64 turns, seed 100. Nothing learned (goal rate 0.12–0.25).
- `main.sh`, `severity.sh`, `e7.sh`, `controls.sh`, `followup.sh`, `review.sh`: the reported experiments. `mechanism.sh` is the first, unpaired version of the mechanism controls.

The run folders under `runs/` from these stages are kept, including the failed ones.
