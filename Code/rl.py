#!/usr/bin/python
# 16-831 Fall 2017
# Project 4
# RL questions:
# Fill in the various functions in this file for Q3.2 on the project.

import numpy as np

import gridworld 


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
  Q3.2.1
  This implements value iteration for learning a policy given an environment.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (np.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """

  ## YOUR CODE HERE ##
  raise NotImplementedError()


def policy_from_value_function(env, value_function, gamma):
  """
  Q3.2.1/Q3.2.2
  This generates a policy given a value function.
  Useful for generating a policy given an optimal value function from value
  iteration.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    value_function: np.ndarray
      Optimal value function array of length nS
    gamma: float
      Discount factor, must be in range [0, 1)

  Output:
    np.ndarray
    policy: Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
  """

  ## YOUR CODE HERE ##
  raise NotImplementedError()


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
  """
  Q3.2.2: BONUS
  This implements policy iteration for learning a policy given an environment.

  You should potentially implement two functions "evaluate_policy" and 
  "improve_policy" which are called as subroutines for this.

  Inputs:
    env: environment.DiscreteEnvironment (likely gridworld.GridWorld)
      The environment to perform value iteration on.
      Must have data members: nS, nA, and P
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (np.ndarray, iteration)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """

  ## BONUS QUESTION ##
  ## YOUR CODE HERE ##
  raise NotImplementedError()


if __name__ == "__main__":

  env = gridworld.GridWorld(map_name='8x8')

  # Play around with gamma if you want!
  gamma = 0.9
  # Q3.2.1
  Vs, n_iter = value_iteration(env, gamma)
  policy = policy_from_value_function(env, Vs, gamma)