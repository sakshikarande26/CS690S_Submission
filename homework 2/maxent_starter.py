import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D


def build_trans_mat_gridworld():
  # 5x5 gridworld laid out like:
  # 0  1  2  3  4
  # 5  6  7  8  9 
  # 9  10 11 12 13
  # 14 15 16 17 18
  # 20 21 22 23 24
  # where 24 is a goal state that always transitions to a 
  # special zero-reward terminal state (25) with no available actions
  trans_mat = np.zeros((26,4,26))
  
  # NOTE: the following iterations only happen for states 0-23.
  # This means terminal state 25 has zero probability to transition to any state, 
  # even itself, making it terminal, and state 24 is handled specially below.
  
  # Action 0 = down
  for s in range(24):
    if s < 20:
      trans_mat[s,0,s+5] = 1
    else:
      trans_mat[s,0,s] = 1
      
  # Action 1 = up
  for s in range(24):
    if s >= 5:
      trans_mat[s,1,s-5] = 1
    else:
      trans_mat[s,1,s] = 1
      
  # Action 2 = left
  for s in range(24):
    if s%5 > 0:
      trans_mat[s,2,s-1] = 1
    else:
      trans_mat[s,2,s] = 1
      
 # Action 3 = right
  for s in range(24):
    if s%5 < 4:
      trans_mat[s,3,s+1] = 1
    else:
      trans_mat[s,3,s] = 1

  # Finally, goal state always goes to zero reward terminal state
  for a in range(4):
    trans_mat[24,a,25] = 1  
      
  return trans_mat


def build_state_features_gridworld():
  # There are 4 features and only one is active at any given state, represented 1-hot vector at each state, with the layout as follows:
  # 0 0 0 0 0
  # 0 1 1 1 1
  # 0 0 2 0 0
  # 0 0 0 0 0 
  # 0 0 0 0 4
  # And the special terminal state (25) has all zero state features.

  sf = np.zeros((26,4))  
  sf[0,0] = 1
  sf[1,0] = 1
  sf[2,0] = 1
  sf[3,0] = 1
  sf[4,0] = 1
  sf[5,0] = 1
  sf[6,1] = 1
  sf[7,1] = 1
  sf[8,1] = 1
  sf[9,1] = 1
  sf[10,0] = 1
  sf[11,0] = 1
  sf[12,2] = 1
  sf[13,0] = 1
  sf[14,0] = 1
  sf[15,0] = 1
  sf[16,0] = 1
  sf[17,0] = 1
  sf[18,0] = 1
  sf[19,0] = 1
  sf[20,0] = 1
  sf[21,0] = 1
  sf[22,0] = 1
  sf[23,0] = 1
  sf[24,3] = 1
  return sf


           
def calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index):
  """
  Implement steps 1-3 of Algorithm 1 in Ziebart et al.
  
  For a given reward function and horizon, calculate the MaxEnt policy that gives equal weight to equal reward trajectories
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  r_weights: a size F array of the weights of the current reward function to evaluate
  state_features: an S x F array that lists F feature values for each state in S
  term_index: the index of the special terminal state
  
  return: an S x A policy in which each entry is the probability of taking action a in state s
  """
  n_states = np.shape(trans_mat)[0]

  # Step 1: reward only depends on which state you're in, not which action
  # you take there, so this is a size-S vector, not S x A
  state_rewards = state_features.dot(r_weights)
  exp_rewards = np.exp(state_rewards)

  # Step 2: with horizon steps left, Zs is 1 at the terminal state (you've
  # already arrived, zero further return) and 0 everywhere else
  Zs = np.zeros(n_states)
  Zs[term_index] = 1

  for _ in range(horizon):
    # Za[s,a] is exp(r(s)) times the return-weighted mass of wherever
    # taking action a from s can land you
    Za = exp_rewards[:, None] * np.einsum('sak,k->sa', trans_mat, Zs)
    Zs = Za.sum(axis=1)
    # Step 2 correction: a trajectory is also free to simply stay put at the
    # terminal state rather than keep going, so that "already done" option
    # contributes exp(0) = 1 back into Zs at every remaining step
    Zs[term_index] += 1

  # Step 3: the policy is each action's share of its state's total Z. A
  # state Zs could be exactly 0 (e.g. unreachable under trans_mat), which
  # would otherwise turn this division into NaN, so guard it explicitly
  # and leave the policy at 0 there instead of propagating NaNs
  with np.errstate(invalid='ignore', divide='ignore'):
    policy = Za / Zs[:, None]
  policy = np.nan_to_num(policy, nan=0.0, posinf=0.0, neginf=0.0)

  return policy


  
def calcExpectedStateFreq(trans_mat, horizon, start_dist, policy):
  """
  Implement steps 4-6 of Algorithm 1 in Ziebart et al.
  
  Given a MaxEnt policy, begin with the start state distribution and propagate forward to find the expected state frequencies over the horizon
  
  trans_mat: an S x A x S' array of transition probabilites from state s to s' if action a is taken
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  start_dist: a size S array of starting start probabilities - must sum to 1
  policy: an S x A array array of probabilities of taking action a when in state s
  
  return: a size S array of expected state visitation frequencies
  """
  
  n_states = np.shape(trans_mat)[0]

  # Step 4: at t=0 you're exactly where you started
  visitation = np.zeros((horizon, n_states))
  visitation[0] = start_dist

  for t in range(1, horizon):
    # Step 5: probability of landing in s' next is the sum, over every
    # predecessor state/action pair, of "were you there" times "did you take
    # that action" times "did it transition you to s'"
    visitation[t] = np.einsum('sa,sai->i', visitation[t - 1][:, None] * policy, trans_mat)

  # Step 6: total expected visits to each state, summed across the horizon
  state_freq = visitation.sum(axis=0)
  return state_freq
  


def maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index):
  """
  Implement the outer loop of MaxEnt IRL that takes gradient steps in weight space
  
  Compute a MaxEnt reward function from demonstration trajectories
  
  trans_mat: an S x A x S' array that describes transition probabilities from state s to s' if action a is taken
  state_features: an S x F array that lists F feature values for each state in S
  demos: a list of lists containing D demos of varying lengths, where each demo is series of states (ints)
  seed_weights: a size F array of starting reward weights
  n_epochs: how many times (int) to perform gradient descent steps
  horizon: the finite time horizon (int) of the problem for calculating state frequencies
  learning_rate: a multiplicative factor (float) that determines gradient step size
  term_index: the index of the special terminal state
  
  return: a size F array of reward weights
  """
  
  n_states = np.shape(state_features)[0]
  n_demos = len(demos)

  # Empirical feature expectation f~: how often each feature was actually
  # visited in the demonstrations, averaged across demos (the README flags
  # this averaging as the one detail the paper's notation leaves ambiguous)
  demo_feature_counts = np.zeros(np.shape(state_features)[1])
  for demo in demos:
    for state in demo:
      demo_feature_counts += state_features[state]
  demo_feature_counts /= n_demos

  # Empirical start distribution: just where the demos happened to begin
  start_dist = np.zeros(n_states)
  for demo in demos:
    start_dist[demo[0]] += 1
  start_dist /= n_demos

  r_weights = np.copy(seed_weights)  # don't clobber the caller's array

  for _ in range(n_epochs):
    policy = calcMaxEntPolicy(trans_mat, horizon, r_weights, state_features, term_index)
    state_freq = calcExpectedStateFreq(trans_mat, horizon, start_dist, policy)
    expected_feature_counts = state_freq.dot(state_features)

    # Gradient of the demo log-likelihood w.r.t. r_weights: demonstrated
    # feature counts minus what the current reward's policy would produce
    grad = demo_feature_counts - expected_feature_counts
    r_weights += learning_rate * grad

  return r_weights
  
 
 
if __name__ == '__main__':
  
  # Build domain, features, and demos
  trans_mat = build_trans_mat_gridworld()
  state_features = build_state_features_gridworld() 
  demos = [[4,9,14,19,24,25],[3,8,13,18,19,24,25],[2,1,0,5,10,15,20,21,22,23,24,25],[1,0,5,10,11,16,17,22,23,24,25]]
  seed_weights = np.zeros(4)
  term_index = 25
  
  # Parameters
  n_epochs = 25
  horizon = 15
  learning_rate = 0.1
  
  # Main algorithm call
  r_weights = maxEntIRL(trans_mat, state_features, demos, seed_weights, n_epochs, horizon, learning_rate, term_index)
  
  # Construct reward function from weights and state features
  reward_fxn = []
  for s_i in range(25):
    reward_fxn.append( np.dot(r_weights, state_features[s_i]) )
  reward_fxn = np.reshape(reward_fxn, (5,5))
  
  # Plot reward function
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X = np.arange(0, 5, 1)
  Y = np.arange(0, 5, 1)
  X, Y = np.meshgrid(X, Y)
  surf = ax.plot_surface(X, Y, reward_fxn, rstride=1, cstride=1, cmap=cm.coolwarm,
			linewidth=0, antialiased=False)
  plt.show()



