#!/usr/bin/python
# 16-831 Fall 2017
# Project 4
# IRL questions:
# Fill in the various functions in this file for Q3.3 on the project.

import numpy as np
import cvxopt as cvx

import gridworld
import rl


def irl_lp(policy, T_probs, discount, R_max, l1):
  """
  Solves the linear program formulation for finite discrete state IRL.

  Inputs:
    policy: np.ndarray
      Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
    T_probs: np.ndarray
      nS x nA x nS matrix where:
      T_probs[s][a] is a probability distribution over states of transitioning
      from state s using action a.
      Can be generated using env.generateTransitionMatrices.
    gamma: float
      Discount factor, must be in range [0, 1)
    R_max: float
      Maximum reward allowed.
    l1: float
      L1 regularization penalty.

  Output:
    np.ndarray
    R: Array of rewards for each state.
  """

  T_probs = np.asarray(T_probs)
  nS, nA, _ = T_probs.shape

  ## YOUR CODE HERE ##
  # Create c, G and h in the standard form for cvxopt.
  # Look at the documentation of cvxopt.solvers.lp for further details

  # Don't do this all at once. Create portions of the vectors and matrices for
  # different parts of the objective and constraints and concatenate them
  # together using something like np.r_, np.c_, np.vstack and np.hstack.
  raise NotImplementedError()

  # You shouldn't need to touch this part.
  c = cvx.matrix(c)
  G = cvx.matrix(G)
  h = cvx.matrix(h)
  sol = cvx.solvers.lp(c, G, h)

  R = np.asarray(sol["x"][:nS]).squeeze()

  return R


if __name__ == "__main__":

  env = gridworld.GridWorld(map_name='8x8')

  # Generate policy from Q3.2.1
  gamma = 0.9
  Vs, n_iter = rl.value_iteration(env, gamma)
  policy = rl.policy_from_value_function(env, Vs, gamma)

  T = env.generateTransitionMatrices()

  # Q3.3.5
  # Set R_max and l1 as you want.
  R = irl_lp(policy, T, gamma, R_max, l1)

  # You can test out your R by re-running VI with your new rewards as follows:
  # env_irl = gridworld.GridWorld(map_name='8x8', R=R)
  # Vs_irl, n_iter_irl = rl.value_iteration(env_irl, gamma)
  # policy_irl = rl.policy_from_value_function(env_irl, Vs_irl, gamma)