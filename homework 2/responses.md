# Homework 2 Responses

Status: `calcMaxEntPolicy`, `calcExpectedStateFreq`, and `maxEntIRL` are all implemented. All six
questions are answered below (Q1/Q2 numerically, backed by `lr_sweep.png` and `reward_surface.png`
at the repo root; Q3-Q6 conceptually/derivation-only, unchanged since the earlier passes).

## Q1

**Experiment and choose a good static step size / learning rate for gradient descent... Describe how you chose an appropriate learning rate.**

Swept `learning_rate ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}`, each for 100 epochs (horizon 15, seed
weights all zero, the provided demos), tracking `||grad||` per epoch (plot: `lr_sweep.png`,
y-axis log scale). For each rate, checked whether `||grad||` decreased monotonically every epoch
(no oscillation):

| learning_rate | monotonic decrease? | epochs with an increase |
|---|---|---|
| 0.01 | yes | 0 |
| 0.05 | yes | 0 |
| 0.1  | yes | 0 |
| 0.2  | yes | 0 |
| 0.5  | no  | 7 (first at epoch 3) |
| 1.0  | no  | 23 (first at epoch 1) |

**Chosen rate: 0.2** — it's the largest of the swept rates that still converges monotonically
(`||grad||` drops from ~5.4 at epoch 0 to ~0.21 by epoch 100 with no step ever increasing the
gradient norm), so it gets to a low-gradient solution in the fewest epochs without the
overshoot/oscillation seen at 0.5 and 1.0 (where larger steps repeatedly overshoot the minimum and
bounce back, and the final weights end up much larger in magnitude without the gradient actually
being smaller — at `lr=1.0` the max gradient norm during training is even *higher* than the
starting value, 8.28 vs 5.40, a clear overshoot signature). Smaller rates (0.01/0.05/0.1) also
converge monotonically but take more epochs to reach a comparably small gradient norm, so 0.2 is
the efficient choice among the monotonic options.

## Q2

**Run MaxEntIRL for 100 epochs with a horizon of 15 and describe the shape of the reward function... Include a graph of the reward function in your writeup.**

Ran `maxEntIRL(trans_mat, state_features, demos, seed_weights=zeros(4), n_epochs=100, horizon=15, learning_rate=0.2, term_index=25)`
(learning rate from Q1). Recovered weights: `r_weights ≈ [-2.21, -8.70, -3.98, +1.05]` (feature 0 =
open floor, feature 1 = the row-1 band of states 6-9, feature 2 = state 12 alone, feature 3 = the
goal state 24). Reward surface saved to `reward_surface.png`; as a 5x5 grid of `r(s)`:

```
-2.21  -2.21  -2.21  -2.21  -2.21
-2.21  -8.70  -8.70  -8.70  -8.70
-2.21  -2.21  -3.98  -2.21  -2.21
-2.21  -2.21  -2.21  -2.21  -2.21
-2.21  -2.21  -2.21  -2.21  +1.05
```

**Qualitative shape:** everything is negative except the goal, which is the only positive cell —
this is the standard "living cost + goal bonus" shape you'd expect for a shortest-path-style task,
not unusual by itself. What *is* notable is the row-1 band (states 6-9) carries by far the most
negative reward (-8.70, roughly 4x more negative than open floor), while state 12 is penalized more
moderately (-3.98, between open-floor and the row-1 band). Checking the empirical feature counts
`f̃` computed from the demos (`[6.5, 0.5, 0.0, 1.0]`): the demos visit row-1 states only twice across
all 4 trajectories (both near the edges, states 8 and 9) and **never** visit state 12 at all. The
algorithm is doing exactly what MaxEnt IRL is supposed to do — fitting weights so that the policy's
*expected* visitation matches the *empirical* visitation. Since state 12 (dead center of the grid,
and the only state with feature 2) is a natural shortcut for several of the straight-line paths a
pure shortest-path agent would want to take toward the goal, but none of the 4 demonstrated
trajectories ever pass through it, the optimizer has to push its reward down enough to suppress
that shortcut-seeking behavior.

**Is state 12's treatment sensible?** Only partially, and this is worth flagging as a likely
over-fit to a small, idiosyncratic demo set rather than a reflection of the true ground-truth
reward. All 4 demos route along the grid's edges/second-row path rather than through the center —
plausibly just because the demonstrator always started near an edge (`start_dist` is nonzero only
at states 1-4, all top row) and a near-edge start rarely has reason to cut through the center. With
only 4 demonstrations and none passing anywhere near state 12, there's no real evidence either way
about whether state 12 is actually "bad" (e.g. an obstacle) or just "unvisited by coincidence of
where these particular trajectories started" — the -3.98 weight is as easily an artifact of sparse
demo coverage as a learned preference. The row-1 penalty (-8.70) is comparatively better evidenced,
since it's involved in all 4 of the demonstrated paths choosing to route *around* rather than
*through* that band when a given start/goal pair gives them the option (e.g. demo 1 starts at state
4, directly above the row-1 band, and still detours down through column 4 rather than cutting left
through row 1 to reach the goal).

**Cross-check against the `calcExpectedStateFreq` scaling caveat (from the earlier pass):** the
earlier sanity checks showed that with `horizon=15` on this grid, a reward bonus concentrated at a
single state (like the goal) barely shifts the *policy's action ratios* at a given state, because
almost every long trajectory passes through that state at least once regardless of the first
action taken, so the bonus's multiplicative effect is nearly uniform across choices. That same
effect likely explains why the fitted weight on the goal feature (+1.05) is much smaller in
magnitude than the penalty weights (-2.21 to -8.70): since `calcExpectedStateFreq` is already
structurally biased toward reaching/passing the goal even under near-zero goal reward (via the
`Z_s[term_index] += 1` backward-pass term, see Q6), gradient descent doesn't need a large positive
goal weight to make expected goal-visitation match the demos' goal-visitation — most of that work
is already done by the algorithm's built-in terminal-state bias. That means the *relative*
magnitudes recovered here (row-1 band most negative, goal weakly positive) should be trusted more
than the literal numeric values, which are partly an artifact of this horizon/structural effect
rather than purely reflecting the demonstrated preferences.

## Q3

**Assume we have a single demonstration trajectory τ that we will use to compute an estimate the expert's feature expectations f̃. In the simpler case of deterministic dynamics, calculate the gradient of the loss function L(θ), where L(θ) = log P(τ|θ) = log[e^(θᵀf_τ) / Z(θ)]. Recall that Z(θ) = Σ_ζ e^(θᵀf_ζ). The endpoint of your proof should be Eqn 6 in the paper, though it will contain P(ζ|θ) rather than P(ζ|θ,T) due to the simplifying deterministic dynamics assumption.**

(Note: render this with proper LaTeX in the actual PDF writeup — plain-text math here for
reference.)

**Setup.** `f_τ` is the sum of `state_features[s]` over every state `s` visited by trajectory `τ`
(this is exactly what `maxEntIRL` computes before averaging over demos — see
`demo_feature_counts` in `maxent_starter.py:202-206`, which for a single demo is `f_τ`). The loss
is already given in log form:

  L(θ) = log P(τ|θ) = log[e^(θᵀf_τ)] − log Z(θ) = θᵀf_τ − log Z(θ)

splitting the log of a quotient into a difference of logs, and `log(e^x) = x` collapses the
numerator immediately.

**Differentiate term 1 (linear in θ).**

  ∂/∂θ [θᵀf_τ] = f_τ

**Differentiate term 2 (the log-partition term — this is where log/exp/chain rule matter).**

  ∂/∂θ log Z(θ) = (1/Z(θ)) · ∂Z(θ)/∂θ        [chain rule: d(log u)/dθ = (1/u)·du/dθ]

`Z(θ) = Σ_ζ e^(θᵀf_ζ)` is a sum over every possible trajectory ζ, so differentiate term-by-term
under the sum:

  ∂Z(θ)/∂θ = Σ_ζ ∂/∂θ [e^(θᵀf_ζ)]
           = Σ_ζ e^(θᵀf_ζ) · f_ζ          [chain rule: d(e^u)/dθ = e^u · du/dθ, and ∂(θᵀf_ζ)/∂θ = f_ζ]

Dividing each term by Z(θ) turns `e^(θᵀf_ζ)/Z(θ)` back into `P(ζ|θ)` (the deterministic-dynamics
simplification of Eqn 5 — no transition-probability product needed since each action sequence maps
to exactly one trajectory), so:

  ∂/∂θ log Z(θ) = Σ_ζ P(ζ|θ) · f_ζ

This is the *expected* feature count of a trajectory sampled from the current MaxEnt distribution
over all possible trajectories ζ — exactly the quantity `calcExpectedStateFreq` computes in
aggregate, state-indexed form: `Σ_s state_freq(s) · state_features(s)` (`f_expected` /
`expected_feature_counts` in `maxent_starter.py:219`), since `state_freq(s)` is the expected number
of times state `s` is visited, and summing a state's feature vector weighted by its visitation
count over all states is equivalent to summing each full trajectory's feature vector weighted by
that trajectory's probability.

**Putting it together — Eqn 6:**

  ∂L(θ)/∂θ = f_τ − Σ_ζ P(ζ|θ) · f_ζ

(The paper's Eqn 6 writes this as `f̃ − Σ_ζ P(ζ|θ,T) f_ζ`, with `f̃` averaged over multiple
demonstrations and the transition-probability-weighted `P(ζ|θ,T)` for the general stochastic case;
here, with a single demo and deterministic dynamics, it reduces to `f_τ − Σ_ζ P(ζ|θ) f_ζ`.) This is
exactly `grad = demo_feature_counts − expected_feature_counts` in `maxEntIRL`
(`maxent_starter.py:221-224`), and since `L(θ) = log P(τ|θ)` is the log-*likelihood* (not a
negative-log-loss), ascending this gradient (`r_weights += learning_rate * grad`) directly
increases the likelihood of the demonstrated trajectory — matching the `+=` update in the code,
with no sign flip needed.

## Q4

**Compare equations 5 and 7. Intuitively, why does the MaxEnt policy of equation 5 equally weight paths in the example in Figure 2, while the action based distribution model of equation 7 weights them differently? Why is the weighting of equation 7 problematic?**

Answerable now, independent of the remaining stubs. Equation 5 defines probability directly over
whole trajectories as proportional to `exp(reward of trajectory)`, so two trajectories with equal
total reward get exactly equal probability regardless of how many actions/decision points they
pass through. Equation 7 instead builds the trajectory distribution as a product of local
per-state action probabilities (a Markov policy), so a path with more decision points along the
way picks up more multiplicative "chances" to be split in probability mass — paths through states
with more available actions (even if those actions are equally good) get diluted relative to paths
through states with fewer choices. This makes equation 7's distribution sensitive to purely
structural/irrelevant features of the MDP (how many actions happen to be available along a path),
not just to the reward actually earned — exactly the kind of label-bias problem MaxEnt IRL is
designed to avoid, since it means an agent could be modeled as "preferring" a path purely because
it had fewer branching choices, not because it was better.

## Q5

**Explain in your own words what Z_{a_ij} and Z_{s_i} represent over a particular horizon.**

Answerable now, and directly tied to the `calcMaxEntPolicy` implementation
([[README implementation notes]]). `Z_{s_i}` is the (horizon-limited) sum, over all trajectories
starting at state `s_i`, of `exp(return of trajectory)` — i.e. a soft/reward-weighted count of all
the ways you can act from `s_i` onward, within the remaining horizon, including trajectories that
reach the terminal state early (the `+1` term, see Q6). `Z_{a_ij}` is the same quantity restricted
to trajectories that start at `s_i` by taking action `a_j` as the first step — it equals
`exp(r(s_i))` times the `Z_s` values of all possible next states, weighted by the transition
probabilities of taking `a_j` from `s_i`. In the code, `Z_s` is the length-S backward value
computed each horizon iteration and `Z_a` is the S x A array computed from it; `policy[s,a]` is
then just `Z_a[s,a] / Z_s[s]`, i.e. the fraction of `s_i`'s total soft-return mass attributable to
first taking action `a_j`.

## Q6

**Why is 1 added to Z_s for terminal states? (hint: think of the equation for Z_s in terms of returns of trajectories)**

Answerable now — this is exactly the correction implemented in `calcMaxEntPolicy`
([[README implementation notes]]). `Z_s` sums `exp(return)` over all trajectories of up to the
remaining horizon length starting from `s`. A trajectory that has already reached the terminal
state before the horizon is used up doesn't accumulate any further reward (the terminal state has
all-zero features, so `r(terminal) = 0`) and doesn't take any more actions — it just "stays
finished." That still counts as one more valid trajectory of return 0, contributing
`exp(0) = 1` to the sum. Without adding that `1` back in at every iteration, trajectories that
terminate early would be silently dropped from the partition function instead of being counted as
zero-additional-reward continuations, which would bias `Z_s` (and therefore the policy) away from
the terminal/goal state.
