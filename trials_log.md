# Trials and Improvements Log

Chronological record of every run, what changed between runs, and what it taught. Add a row after each run. Final answers go in `responses.md`.

## Part 1: Random policy (`test_gym.py`)

| # | What changed | Result | Takeaway |
|---|---|---|---|
| 1 | First run | Car oscillated, swings grew, eventually reached the flag | Loop never sets `done`, so it ran past 200 steps. Measured: 0 of 200 random episodes reach the flag within 200 steps; 15 of 200 within 5000 steps (median 3818) |

## Part 2: Human driving (`mountain_car_play.py`)

The first `0.0` printed at the start of every run is a startup artifact, not a real episode.

| # | Strategy | Returns | Reached flag | Takeaway |
|---|---|---|---|---|
| 1 | Right, left, right | -146, -162, -168 | 3 of 3 | Works; right alone did not |
| 2 | Left, then right all the way | -200 x5 | 0 of 5 | One left swing stores too little energy |
| 3 | Left, right, left, right | -200, -115, -159, -155, -146, -148 | 5 of 6 | Best and most consistent. Best score -115 |

## Part 3: BC with 1 demo (`mountain_car_bc.py`)

| # | Change from previous | Demo return | Avg / Min / Max | Reached flag | Final loss | Takeaway |
|---|---|---|---|---|---|---|
| 1 | Baseline, first demo | -196 | -187.67 / -200 / -162 | 2 of 6 | 0.15 | Slow demo, narrow data. Low loss but poor control |
| 2 | New demo, slightly faster (I suggested pushing with the velocity and holding each push; not confirmed that this is what was done) | -184 | -161.83 / -174 / -141 | 6 of 6 | 0.35 | Better policy from a better demo. Lower loss did not mean better returns |

Improvements that helped so far:
- Run 2 had a slightly faster demo (-196 to -184) and a higher average (about 26 better). Two runs of 6 evaluations cannot show that the demo caused it.

## Part 4: BC with 5 demos

| # | Change | Demo returns | Avg / Min / Max | Reached flag | Takeaway |
|---|---|---|---|---|---|
| 0 | Rerun of the default `--num_demos 5` (reported in `responses.md`) | -157, -141, -131, -163, -105 | -124.83 / -162 / -85 | 6 of 6 (returns -162, -161, -157, -99, -85, -85) | Best result so far. No timeouts. Final loss 0.58 (highest of the 5-demo runs, so loss does not predict performance). Agent copied the swing strategy |
| 1 | Baseline: `--num_demos 5`, defaults (100 iters, lr 0.1) | (not recorded) | -156.5 / -200 / -98 | 4 of 6 (returns -200, -110, -166, -200, -165, -98) | Final loss 0.40, still high (underfit). Wide spread; 2 timeouts |
| 2 | `--num_bc_iters 500` (assumed; confirm) | (not recorded) | -140.5 / -200 / -83 | 5 of 6 (returns -83, -159, -166, -200, -84, -151) | Better average and best run; 1 timeout left. Final loss not recorded |
| 3 | `--num_demos 5 --num_bc_iters 500` (6 evals, default) | (not recorded) | -148.33 / -158 / -100 | 6 of 6 (returns -158 x5, -100) | No timeouts, very consistent. Loss plateaued near 0.46 by iteration ~70, so extra iterations did not help. Average slightly worse than row 2 (-140.5) but row 2 had a timeout |

## Part 5: Good and bad demo

| # | Change | Demo returns | Avg / Min / Max | Takeaway |
|---|---|---|---|---|
| 1 | `--num_demos 2`: demo 1 hold right (bad), demo 2 good; defaults | -200, -120 | -196.0 / -200 / -176 | Policy fails: 5 of 6 timeouts. Final loss 0.45. Bad demo is about 62% of the data |

## Part 7: BCO(0)

| # | Change (network size, iterations, data) | Inverse dynamics accuracy on demos | Policy Avg / Min / Max | Takeaway |
|---|---|---|---|---|
| 1 | 4-d input (state, change in state), 64-64 net, 1000 iters, 50 random episodes. Tested on scripted demos, not human demos | 90%, 95%, 82% (3 seeds); 100% on random data | -175 / -152 / -157 (avg over 10 evals per seed) | Perfect on random data but 82-95% on demos |
| 2 | 2-d input (position, change in velocity), otherwise same | 91%, 93%, 87% | -170 / -144 / -170 | Similar accuracy, simpler input, kept this version |
| 3 | Real run, 1 human demo (-172), defaults (1000 inv. dyn. iters, 100 BC iters) | 85.5% (100% on random data) | incomplete: first 2 evals -200, -200 (stopped with Ctrl+Z) | Final BC loss 0.52 |
| 4 | Real run, 1 human demo (-162), same settings | 92.6% (100% on random data) | incomplete: first 4 evals all -200 (output cut off) | Final BC loss 0.48 |
| 5 | Real run, 1 human demo (-176), `--num_evals 3`, otherwise defaults | 84.1% (100% on random data) | -181.33 / -200 / -172 (returns -200, -172, -172) | Complete run. 2 of 3 reached the flag. Final BC loss 0.61 (noisy labels) |
| 6 | Random data with each action held 1 to 30 steps (`collect_held_random_data`), scripted demos, 3 seeds | 100%, 100%, 100% (random data range now -1.20 to 0.50) | -165 / -140 / -131 | Fixes the coverage problem. Hold test over 5 seeds: max hold 1 gives 91.5%, 10 gives 99.0%, 30 gives 100% |
| 7 | Real run with held random actions (max hold 30), 1 human demo (-175), `--num_evals 3` | 100% (99.97% on random data) | -142.67 / -144 / -141 (returns -144, -141, -143; max line cut off in paste, -141 inferred) | 3 of 3 reached the flag. Final BC loss 0.37. Best BCO result so far |
| diagnosis | | | | Wrong labels are at positions below -0.9 (and a few at 0.2 to 0.4). Random data covers only positions -0.8 to -0.26 (500 episodes: -0.92 to -0.13), so the demos leave the range the model was trained on |

## Part 5 simulation check (scripted demos, 10 seeds x 20 evals)

| Training data | Avg return | Timeout rate | Predict "right" on train states |
|---|---|---|---|
| Good only (121 samples) | -133.0 +/- 4.8 | 11% | 67% |
| Bad + good (321 samples) | -197.5 +/- 5.2 | 96% | 89% |

Script lives in the session scratchpad (`part5_sim.py`), not in the repo.
