#!/usr/bin/python
# 16-831 Fall 2017
# Project 4
# IRL questions:
# Fill in the various functions in this file for Q3.3 on the project.

import numpy as np
import cvxopt as cvx
from cvxopt import matrix, solvers

import gridworld
import rl
import sys

def getExpectedFuture(a, s, T_probs, policy, discount, nS):
  forward = T_probs[int(policy[s])][s] - T_probs[a][s]
  inv = np.linalg.inv(np.eye(nS) - discount * T_probs[int(policy[s])])
  return np.dot(forward, inv)

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

  # Reshaping T_probs to simplify call to get Pa for a given action
  # The format is now A, S, S instead of S, A, S ;D
  T_probs = np.transpose(T_probs, (1, 0, 2))
  A = set(range(nA))

  # The G Matrix has the following shape:
  #           64               64             64
  # 192:  [  -futurePayoff,    identity1,     zero1     ]
  # 192:  [  -futurePayoff,    zero1,         zero1     ]
  # 64:   [  -identity2,       zero2,       -identity2  ]
  # 64:   [   identity2,       zero2,       -identity2  ]
  # 64:   [  -identity2,       zero2,         zero2     ]
  # 64:   [   identity2,       zero2,         zero2     ]

  # Each row corresponds to the following constraint:
  # -futurePayoff * R + M <= 0 aka -futurePayoff * R >= M
  # -futurePayoff * R <= 0 aka futurePayoff * R >= 0
  # -R - mu <= 0 aka R >= -mu
  # R - mu <= 0 aka R <= mu
  # -R <= R_max aka R >= -R_max
  # R <= R_max

  futurePayoff = np.vstack([
    getExpectedFuture(a, s, T_probs, policy, discount, nS)
    for s in range(nS)
    for a in A - {int(policy[s])}
  ])

  # 192 x 64 Identity Matrix
  identity1 = np.vstack([
    np.eye(1, nS, s)
    for s in range(nS)
    for a in A - {int(policy[s])}
  ])
  # 64 x 64 Identity Matrix
  identity2 = np.eye(nS)

  # 192 x 64 Zero Matrix
  zero1 = np.zeros((nS * (nA - 1), nS))

  # 64 x 64 Zero Matrix
  zero2 = np.zeros((nS, nS))

  G = np.vstack([
      np.hstack([ -futurePayoff,  identity1,  zero1       ]), # -futurePayoff * R >= M
      np.hstack([ -futurePayoff,  zero1,      zero1       ]), # futurePayoff * R >= 0
      np.hstack([ -identity2,     zero2,      -identity2  ]), # R >= -mu
      np.hstack([ identity2,      zero2,      -identity2  ]), # R >= -mu
      np.hstack([ -identity2,     zero2,      zero2       ]), # R >= -R_max
      np.hstack([ identity2,      zero2,      zero2       ])  # R <= R_max
    ])

  c = np.hstack([
      np.zeros(nS),     # Corresponding to R
      -1 * np.ones(nS), # Corresponding to M
      l1 * np.ones(nS)  # Corresponding to mu
    ])


  h = np.vstack([
      np.zeros(((G.shape[0] - 2 * nS), 1)), # the non-R_max part of G
      R_max * np.ones((nS, 1)),             # R >= -R_max
      R_max * np.ones((nS, 1))              # R <= R_max
    ])

  ## YOUR CODE HERE ##
  # Create c, G and h in the standard form for cvxopt.
  # Look at the documentation of cvxopt.solvers.lp for further details

  # Don't do this all at once. Create portions of the vectors and matrices for
  # different parts of the objective and constraints and concatenate them
  # together using something like np.r_, np.c_, np.vstack and np.hstack.
  # raise NotImplementedError()

  # You shouldn't need to touch this part.
  c = cvx.matrix(c)
  G = cvx.matrix(G)
  h = cvx.matrix(h)
  sol = cvx.solvers.lp(c, G, h)

  R = np.asarray(sol["x"][:nS]).squeeze()

  return R

def printGridWorld(title, printArray, width, height, isInt):

  sys.stdout.write("\n\n" + title + "\n\n");
  for i in range(height):
    for j in range(width):
      if (isInt):
        sys.stdout.write("%6s" % str('%d' % printArray[(i * height) + j]) + " ");
      else:
        sys.stdout.write("%6s" % str('%02.2f' % printArray[(i * height) + j]) + " ");
    sys.stdout.write("\n\n");
  sys.stdout.flush();

if __name__ == "__main__":

  env = gridworld.GridWorld(map_name='8x8')

  # Generate policy from Q3.2.1
  gamma = 0.9
  Vs, n_iter = rl.value_iteration(env, gamma)
  policy = rl.policy_from_value_function(env, Vs, gamma)

  T = env.generateTransitionMatrices()

  # Q3.3.5
  # Set R_max and l1 as you want.
  R_max = 1
  l1 = 0.5
  R = irl_lp(policy, T, gamma, R_max, l1)

  printGridWorld("IRL-generated Rewards", R, 8, 8, False)

  # You can test out your R by re-running VI with your new rewards as follows:
  env_irl = gridworld.GridWorld(map_name='8x8', R=R)
  Vs_irl, n_iter_irl = rl.value_iteration(env_irl, gamma)
  policy_irl = rl.policy_from_value_function(env_irl, Vs_irl, gamma)

  printGridWorld("Values for Value Iteration with IRL-Generated Rewards", Vs_irl, 8, 8, False);
  printGridWorld("Policy (Actions) for Value Iteration with IRL-Generated Rewards", policy_irl, 8, 8, True);
  