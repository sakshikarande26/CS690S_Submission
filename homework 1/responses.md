# CS 690S HW1 Responses

Running log of questions and answers. Fill in numbers after each run, then transcribe into the PDF report.

---

## Part 1: Random policy on MountainCar

**Question:** Run `python test_gym.py`. What do you notice happening? What specifically about MountainCar makes the problem difficult for RL?

**Observation (from running it):**
The car oscillates. It rolls back up the left hill, then swings right, and repeats this many times. Each swing is slightly larger than the last, and eventually the car has built up enough momentum to reach the flag.

**Note on the script:** `test_gym.py` sets `done = False` and never updates it, so the loop does not stop when the episode ends. The 200-step limit only sets `truncated=True`, and the loop keeps stepping past it. I measured this with a random policy that ignores the 200-step limit (200 episodes, up to 5000 steps each): 0 of 200 reached the flag within 200 steps, and only 15 of 200 reached it at all within 5000 steps (median 3818 steps among those). So the car reaching the flag in the window only happened because the episode was never cut off.

**Answer:**
The random policy picks left, none, or right uniformly, so its actions mostly cancel out. The car jiggles near the bottom of the valley and rarely builds momentum. Any progress is a random-walk accident. Within the 200-step limit it never reached the flag in my test (0 of 200 episodes), so the return is -200 every episode.

MountainCar is hard for RL for these reasons:
1. **Sparse, uninformative reward.** The reward is -1 per step regardless of what the car does. The agent gets no signal until it first reaches the flag, and a random policy almost never does within 200 steps. Without a success, there is no gradient of reward to learn from.
2. **Exploration problem.** Reaching the goal needs a coordinated sequence of actions: push left to climb the opposite hill, then right, building momentum. Undirected random exploration will not find this reliably.
3. **Engine is too weak to drive straight up.** Accelerating right the whole time does not work. The car must first move away from the goal, which looks like a worse move under a greedy or myopic view of the reward.

---

## Part 2: Human demonstrations

**Question:** Drive the car yourself with `python mountain_car_play.py`. Which strategy do you like best? What is the best score you can get as a human demonstrator?

**Scores per episode:**
- Strategy 1 (right, left, right): 0.0 (spurious first line, appears at the start of every run), -146, -162, -168. Pressing only right did not reach the flag; going left first and then right did.
- Strategy 2 (left, then right all the way up): 0.0 (same startup artifact), then -200 on all 5 episodes. Never reached the flag. One left swing does not build enough momentum.
- Strategy 3 (left, right, left, right): 0.0 (startup artifact), then -200 (first try, timed out), then -115, -159, -155, -146, -148. Flag reached in 5 of 6 real episodes.

**Strategies tried:**
1. Right, left, right: -146, -162, -168 (3 of 3 reached the flag)
2. Left, right: -200 x5 (0 of 5)
3. Left, right, left, right: -115, -159, -155, -146, -148 (5 of 6, first attempt timed out at -200)

**Best score:** -115 (strategy 3)
**Preferred strategy and why:** Strategy 3. It was the best and the most consistent after the first attempt: scores between -115 and -159, with about one cycle of left-right-left-right needed to reach the flag. More swings build more momentum. (The scores for strategies 1 and 3 overlap, so I cannot say strategy 3 is clearly faster than strategy 1.) Strategy 2 fails because a single left swing does not store enough energy to climb the right hill.

---

## Part 3: Behavioral cloning, 1 demo

**Command:** `python mountain_car_bc.py` (1 demonstration, 100 BC iterations, 6 evaluation episodes)

| | Run 1 | Run 2 |
|---|---|---|
| Demo return | -196 | -184 |
| Final BC loss | 0.15 | 0.35 |
| Evaluation returns | -162, -200, -164, -200, -200, -200 | -142, -170, -141, -174, -170, -174 |
| Average | -187.67 | -161.83 |
| Min | -200 | -174 |
| Max | -162 | -141 |
| Reached flag | 2 of 6 | 6 of 6 |

**Demo description:** The car started near the bottom. The first left push did little because the car had almost no speed. Then right, left, right built momentum and the car reached the flag. Both demos were slow (-196 and -184), compared with my best human score of -115.

**Does it imitate well? Does it get stuck?**
Only partly, and it depended on the demo. In run 1 the policy reached the flag in 2 of 6 evaluations (-162 and -164, both faster than the -196 demo) and timed out in the other 4 (did not reach the flag within 200 steps, return -200). The return alone does not show why; I did not record what the car was doing in those runs. In run 2 the policy reached the flag in all 6 evaluations, with an average of -161.83, again faster than its demo (-184).

Two things explain the weak and variable results:
1. **One demonstration is very little data.** It gives roughly 190 (state, action) pairs covering a narrow part of the state space. Evaluation starts from different random initial positions, so the policy often meets states the demo never visited and picks poor actions. Errors then compound (distribution shift).
2. **Low training loss does not mean good control.** Run 1 had a lower final loss (0.15) than run 2 (0.35) but performed worse. The policy fit the single demo tightly without generalizing.

Caveat: the two runs used different demonstrations and only 6 evaluations each, so the comparison is not controlled. The result shows that a single demo's quality and coverage strongly affect BC.

---

## Part 4: Behavioral cloning, 5 demos

**Command:** `python mountain_car_bc.py --num_demos 5` (default 100 BC iterations, 6 evaluation episodes)

**Demo returns (reported run):** -157, -141, -131, -163, -105 (average -139.4)

| | 1 demo (Part 3, run 1) | 1 demo (Part 3, run 2) | 5 demos (default, reported run) | 5 demos (default, earlier run) | 5 demos, 500 iters? (run A) | 5 demos, 500 iters (run B) |
|---|---|---|---|---|---|---|
| Evaluation returns | -162, -200, -164, -200, -200, -200 | -142, -170, -141, -174, -170, -174 | -162, -161, -157, -99, -85, -85 | -200, -110, -166, -200, -165, -98 | -83, -159, -166, -200, -84, -151 | -158, -158, -158, -100, -158, -158 |
| Average | -187.67 | -161.83 | -124.83 | -156.5 | -140.5 | -148.33 |
| Min | -200 | -174 | -162 | -200 | -200 | -158 |
| Max | -162 | -141 | -85 | -98 | -83 | -100 |
| Timeouts | 4 of 6 | 0 of 6 | 0 of 6 | 2 of 6 | 1 of 6 | 0 of 6 |
| Final BC loss | 0.15 | 0.35 | 0.58 | 0.40 | not recorded | 0.46 |

The reported result for this part is the default 5-demo run with known demo returns: average -124.83, min -162, max -85. The other columns are additional trials (see `trials_log.md`), which I include to show the run-to-run variation. For run A I did not record the command; I assumed `--num_bc_iters 500`.

**Did it perform better?**
Yes. With 5 demonstrations the policy averaged -124.83 and reached the flag in all 6 evaluation episodes, compared with -187.67 and -161.83 for the single-demo policies (which timed out in 4 and 0 of 6 episodes). Its best episodes, -85, were also faster than anything the single-demo policies achieved (best -141). Across all my 5-demo runs the averages ranged from -125 to -157, still better than the single-demo averages, but the spread between runs (about 30) is large for only 6 evaluations from random start positions, so I would not trust the exact ranking of individual runs. The earlier default run, for example, timed out in 2 of 6 episodes.

**Why?**
More demonstrations cover more of the state space (different positions and velocities along the swing, and different starting points), so the policy meets fewer unfamiliar states at test time and compounding error is smaller. A single demo gives roughly 190 samples from one trajectory; five demos give roughly 700 (the five demo returns correspond to about 700 steps in total).

Performance is also limited by the model and the input. The BC loss stayed high (0.40 to 0.58) in every 5-demo run and did not predict performance: the run with the highest loss (0.58) had the best returns. In the 500-iteration run the loss flattened out by about iteration 70, so extra training did not help. The policy network is small (8 hidden units) and its input is unnormalized: velocity is only about +/-0.07 while position spans about 1.8, so the network barely sees velocity, which is what decides which way to push. Demos are also noisy: human timing differs slightly between swings, so the same state can map to different actions, and the cross-entropy loss cannot reach zero.

**Does the agent copy your strategy?**
Yes. Watching the evaluation episodes, the agent imitates my strategy: it swings left, then right, then left again, building momentum the way my demonstrations did. In the reported run all 6 episodes reached the flag, with returns between -85 and -162, similar to my demonstrations (-105 to -163).

In earlier 5-demo runs the agent timed out in some episodes (for example 2 of 6 in the earlier default run). I did not record what the agent was doing in those episodes, so the following is the likely explanation, not an observation. BC learns a mapping from state to action, and it only sees states that appear in the demonstrations. When the car's trajectory drifts slightly from the demonstrated one (a different starting position, or a swing that is a little too small), it enters states the demos barely cover. There the policy can choose a wrong action, such as pushing against the car's motion, which loses energy and moves it further from the demonstrated states. These errors compound over the episode (distribution shift), and the car runs out of time. Human demonstrations are also slightly inconsistent in timing, so the same state can have different labels, which blurs the learned decision boundary near the points where the car should switch direction. Having 5 demos covers more states and reduces this problem, but with a small network and only 100 training iterations it does not remove it.

---

## Part 5: Good and bad demonstrations

**Command:** `python mountain_car_bc.py --num_demos 2` (demo 1: hold right the whole episode; demo 2: a good demo; defaults, 100 BC iterations)

**Demo returns:** -200 (bad, held right), -120 (good)

**Evaluation returns (6 runs):** -200, -200, -200, -200, -176, -200
**Average / Min / Max return:** -196.0 / -200 / -176 (final BC loss 0.45)

**What does the policy learn?**
It mostly learns to push right. The policy timed out in 5 of 6 evaluations and only one episode reached the flag (-176). That is much worse than the good-demo results (averages of -162 and -188 with 1 good demo, and -125 to -157 with 5 good demos). The bad demo lasted all 200 steps, while the good demo took only 120 steps, so about 62% of the 320 training samples say "push right" in every state, including states where the right action is to push left to build momentum. At each state the two demos give conflicting labels, and the classifier ends up favoring the more common one. The final loss stayed high (0.45), which fits the demos disagreeing with each other. I did not check the learned action probabilities directly. The single successful episode was probably a favorable start position where pushing right was enough, but I did not verify this.

**Supporting simulation (scripted demos, not my own data):** To check the mechanism, I trained the same BC code (default network, 100 iterations) on scripted demos over 10 random seeds, evaluating each policy on 20 episodes. The scripted good demo pushes in the direction of the velocity (return -122, 122 steps); the scripted bad demo holds right (return -200, 200 steps).

| Training data | Avg return (mean +/- std over seeds) | Timeout rate | Training states where policy predicts "right" |
|---|---|---|---|
| Good demo only (121 samples) | -133.0 +/- 4.8 | 11% | 67% |
| Bad + good (321 samples) | -197.5 +/- 5.2 | 96% | 89% |

Adding the bad demo moves the policy from usually solving the task to almost never solving it, which matches the real run (-196.0 average). This supports the explanation above, but it is a simulation with perfect scripted demos, so the exact numbers will differ from human demos.

**Why might bad demonstrations be a problem?**
BC treats every (state, action) pair as correct and has no notion of demonstration quality. A bad demo adds wrong labels, and because it is as long as or longer than the good one (a failed episode runs the full 200 steps), it can outweigh the good demo. The conflicting labels also cannot be reconciled by a small network, so the policy ends up wrong in the states that matter most. More generally, BC copies the demonstrator's mistakes, and it cannot tell from the data alone which demos are good.

**One idea for making BC robust to a minority of bad demos:**
Use the return of each demonstration to filter or weight the data. For example, discard (or down-weight in the loss) demos whose total return is far below the others, such as dropping any demo that timed out when most demos reached the flag. This works as long as bad demos are a minority, because the median or typical return then identifies the good behavior. Another option is to train an ensemble of policies on different subsets of demos and keep only the state-action labels that most of them agree on.

---

## Part 6: BCO(0), conceptual changes to the BC code

(Answered before starting Part 7.)

BC needs the demonstrator's actions as labels. BCO(0) uses only the demonstrator's states (the (s, s') pairs), so the actions must be estimated. The changes to the BC code are:

1. **Stop using the demonstrator's actions.** The demos still provide the sequence of states, but the key presses are discarded and never used for training.
2. **Collect random interaction data.** Run a random policy in the environment and record (s, a, s') tuples. Here the actions are known, since we chose them. (`collect_random_interaction_data` in the starter code does this.)
3. **Learn an inverse dynamics model.** Train a classifier that takes (s, s') and predicts the action a that caused the transition. It is trained by supervised learning (cross-entropy) on the random data only. In MountainCar the input is 4-dimensional (position and velocity before and after) and the output is 3 action logits.
4. **Label the demos with the model.** For each consecutive pair (s, s') in the demonstrations, predict the action with the inverse dynamics model, using the argmax as the estimated action.
5. **Run ordinary BC on the labeled demos.** Train the policy on (s, estimated action) with the same cross-entropy loss as before. Policy evaluation is unchanged.

The "(0)" means no post-demonstration interaction: the inverse model is learned once, before BC, from the random data, and is not refined afterwards. The full BCO(alpha) version would run the learned policy in the environment to collect more (s, a, s') data, retrain the inverse model on it, and repeat, with alpha controlling how much post-demonstration data is collected. In the starter code, this means replacing the ground-truth actions passed to `train_policy` with the output of the inverse dynamics function (the TODO in `mountain_car_bco.py`).

Note: checked against a summary of the BCO paper (Torabi et al., 2018). In the paper, the number of post-demonstration interactions per iteration is M = alpha * (number of pre-demonstration interactions), so alpha = 0 means no post-demonstration interaction, and the inverse model is trained on the pre-demonstration random exploration data.

---

## Part 7: BCO(0) implementation

**What was implemented (in `mountain_car_bco.py`):**
- `InverseDynamicsNetwork`: a classifier with two hidden layers of 64 units and 3 output logits. Its input is the position and the change in velocity: an action only changes the velocity by about 0.001, and the size of that change also depends on the position (gravity). Inputs are standardized using the mean and standard deviation of the training data.
- `collect_held_random_data`: random interaction data in which each random action is repeated for 1 to 30 steps.
- `inverse_dynamics(...)`: trains the model on the random data only (Adam, lr 0.01, 1000 iterations, 50 random episodes), then labels each (state, next state) pair in the demonstration. The demonstrator's true actions are only used to print accuracy; the policy is trained on the estimated actions.

**What I tried, and what I found:**
1. *First version.* The random data came from `collect_random_interaction_data` (a new random action every step). Accuracy was 100% on the random data but only 84% to 93% on the demonstrations (three real runs: 85.5%, 92.6%, 84.1%). Changing the input (state and change in state, 4-d, against position and change in velocity, 2-d) made little difference (scripted demos: 82% to 95% against 87% to 93%), so I kept the simpler 2-d input.
2. *Diagnosis.* On a scripted demo, the wrong labels were at positions below about -0.9, outside the range of the random data. A random policy that changes action every step jiggles near the valley floor and only covers positions from -0.8 to -0.26 (50 episodes), and even 500 episodes reached only -0.92. The model is accurate where it was trained and wrong where it was not. Training longer could not fix this, because the training loss was already near zero.
3. *Fix.* Hold each random action for several steps so the car swings up the hills. On scripted demos (5 seeds), accuracy on demos was 91.5% with no holding, 99.0% with holds up to 10 steps, and 100% with holds up to 30 steps. Random-data positions then covered -1.20 to 0.50.

**Inverse dynamics accuracy on the demonstration data** (training on random data only):

| Random data | Accuracy on random data | Accuracy on my demonstration |
|---|---|---|
| New action every step | 100% | 85.5%, 92.6%, 84.1% (three runs) |
| Actions held up to 30 steps | 99.97% | 100% |

**Resulting policy (1 human demo of -175, 100 BC iterations, 3 evaluations):** returns -144, -141, -143, average -142.67, min -144, max -141 (the max is taken from the three returns; its printed line was cut off). All 3 evaluations reached the flag. With the earlier, less accurate model (84.1% on demos) the same setup gave -200, -172, -172 (average -181.33). For comparison, plain BC with 1 demo gave -187.67 and -161.83 (Part 3).

**Caveats:** these are single runs with 3 evaluations, so the improvement over BC is encouraging but not statistically established. The hold-length comparison was done on scripted demos, not human ones. I did not tune the number of inverse dynamics iterations (1000, the default) or the network size beyond the first choice, because the accuracy problem was data coverage, not training.
