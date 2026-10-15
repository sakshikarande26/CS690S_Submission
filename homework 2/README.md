# Homework 2 for CS 690S AI Alignment

### Overview

In this assignment, you will be implementing maximum entropy inverse reinforcement learning, as described by Ziebart et al. (2008) in``Maximum Entropy Inverse Reinforcement Learning".

**NOTE: There are two typos in Algorithm 1 of that paper:**
- In step 4, instead of D_{s_i, t}, it should be: 
  - ![image](https://github.com/sniekum/MaxEntIRL_assignment/assets/1664131/e7c49e02-84b1-4229-b6b4-9b27812aedb6)

- In step 5, it should be:
  - <img width="907" height="107" alt="Screenshot 2026-09-28 at 4 37 30 PM" src="https://github.com/user-attachments/assets/e9e1cb9e-4f33-4892-8355-1f259c47cdb1" />
  - In other words, the probability of a state s_i at the next timestep is based on how often you started in all the states s_k that you could have come from, multiplied by the appropriate action and transition probabilities that would have taken you from s_k to s_i

Your task is to write three functions, as specified in the provided file.  The three functions are:

1. **calcMaxEntPolicy(trans\_mat, horizon, r\_weights, state\_features):**
  
   - Description: For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal return trajectories
  
   - Parameters:
     - trans_mat: an S x A x S' array of transition probabilities from state s to s' if action a is taken
     - horizon: the finite time horizon (int) of the problem for calculating state frequencies
     - r_weights: a size F array of the weights of the current reward function to evaluate
     - item state_features: an S x F array that lists F feature values for each state in S
     - item term_index: the index of the special terminal state
  
   - Return: an S x A policy in which each entry is the probability of taking action a in state s


2. **calcExpectedStateFreq(trans\_mat, horizon, start\_dist, policy):**
  
   - Description: Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
   - Parameters:
     - trans_mat: an S x A x S' array of transition probabilities from state s to s' if action a is taken
     - horizon: the finite time horizon (int) of the problem for calculating state frequencies
     - start_dist: a size S array of starting start probabilities - must sum to 1
     - policy: an S x A array array of probabilities of taking action a when in state s
  
    - Return: a size S array of expected state visitation frequencies


3. **maxEntIRL(trans\_mat, state\_features, demos, seed\_weights, n\_epochs, horizon, learning\_rate):**
  
   - Description: Compute a MaxEnt reward function from demonstration trajectories.  This will invoke the above two functions.

   - Parameters: 
     - trans_mat: an S x A x S' array that describes transition probabilities from state s to s' if action a is taken
     - state_features: an S x F array that lists F feature values for each state in S
     - demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
     - seed_weights: a size F array of starting reward weights
     - n\_epochs: how many times (int) to perform gradient descent steps
     - horizon: the finite time horizon (int) of the problem for calculating state frequencies
     - learning_rate: a multiplicative factor (float) that determines gradient step size
     - term_index: the index of the special terminal state

   - Return: a size F array of reward weights


The transition matrix for a 5x5 gridworld domain is provided in the Python starter code, with states laid out as follows:

```
0  1  2  3  4
5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
20 21 22 23 24
```


24 is a goal state that always transitions to a special terminal state, 25.  There are 4 actions: up, down, left, and right, which move the agent accordingly (and deterministically), unless the agent would move past the edge of the world, in which it stays in place.  All actions at state 24 lead to state 25 deterministically.  All actions at state 25 lead to each possible next state with probability zero, making it terminal. 

There are 4 features and only one is active at any given state, represented 1-hot vector at each state, with the layout as follows:
```
0  0  0  0  0 
0  1  1  1  1 
0  0  2  0  0 
0  0  0  0  0 
0  0  0  0  3 
```
The special terminal state (25) has all zero state features.

**Notes:**

- $\tilde{\textrm{f}}$ should be averaged over all demonstrations trajectories (i.e. it is an expected value that needs to be divided by the number of trajectories).  The paper isn't totally clear about this.
- For all questions, seed the reward weights with all zeros unless instructed otherwise.

## Implementation notes

### calcMaxEntPolicy

Backward recursion (steps 1-3 of Algorithm 1), fully vectorized over states/actions, looped only over `horizon`:

- `rewards = state_features @ r_weights` (size S), `exp_r = exp(rewards)`.
- `Z_s` initialized to zeros except `Z_s[term_index] = 1`.
- Each of `horizon` iterations:
  - `Z_a[s,a] = exp_r[s] * sum_k trans_mat[s,a,k] * Z_s[k]` via `np.einsum('sak,k->sa', trans_mat, Z_s)`.
  - `Z_s[s] = sum_a Z_a[s,a]`.
  - `Z_s[term_index] += 1`.
- Final policy: `policy[s,a] = Z_a[s,a] / Z_s[s]`, with rows where `Z_s[s] == 0` zeroed out instead of propagating NaN/inf.

**Paper corrections applied:**
- Step 4 (summing into `s_i` from predecessor states `s_k`) applies to `calcExpectedStateFreq`'s forward pass, not this function — noted here for context since both corrections come from the same README callout.
- Step 5's `+1` is added to `Z_s[term_index]` inside the loop, not just once at the end — this is the correction applied in this function. It represents the contribution of trajectories that have already terminated (return 0 onward) to the partition function, each additional horizon step.

**Shapes:** `trans_mat` is S x A x S', `Z_s` is size S, `Z_a` is S x A, `policy` is S x A.

**Sanity checks run (not committed as a script):** with `r_weights` all zero, `trans_mat`/`state_features` from `build_trans_mat_gridworld`/zeroed features, `horizon=15`, `term_index=25`:
- All 25 non-terminal rows of `policy` sum to 1 (`np.allclose(policy[:25].sum(axis=1), 1.0)` → True).
- The terminal state's row (`policy[25]`) is all zero.
- Interior states (e.g. 7, 12) are approximately uniform across actions (~0.22-0.30 per action rather than exactly 0.25, because `Z_s[term_index] += 1` each horizon step still pulls weight toward paths that reach the terminal state sooner, even with zero reward).

