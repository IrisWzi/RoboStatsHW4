#!/usr/bin/python
# 16-831 Fall 2017
# Project 4
# RL questions:
# Fill in the various functions in this file for Q3.2 on the project.

import gridworld
import numpy as np
import sys

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


  iteration = 0
  value_function = np.zeros(env.nS) # Initiliazes all v's to 0
  value_function.fill(tol)

  for iteration in range(max_iterations):
      delta = 0
      for state in range(env.nS):
        v = value_function[state];
        maxValue = -1 * float("inf");
        actionForMaxValue = -1;
        for action in range(env.nA):
          currValue = 0;
          for nextStateIndex in range(len(env.P[state][action])):
            currP = env.P[state][action][nextStateIndex]
            currProbability = currP[0]
            currNextState = currP[1]
            currReward = currP[2]
            currValue += (currProbability * (currReward + gamma * value_function[currNextState]));

          if (currValue >= maxValue):
            maxValue = currValue
            actionForMaxValue = action

        value_function[state] = maxValue
        delta = max(delta, abs(v - value_function[state]))

      if delta < tol:
        break

  return (value_function, iteration)


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

  policy = np.zeros(env.nS)
  policy.fill(-1);
  for state in range(env.nS):
    v = value_function[state];
    maxValue = -1 * float("inf");
    actionForMaxValue = -1;
    for action in range(env.nA):
      currValue = 0;
      for nextStateIndex in range(len(env.P[state][action])):
        currP = env.P[state][action][nextStateIndex]
        currProbability = currP[0]
        currNextState = currP[1]
        currReward = currP[2]
        currValue += (currProbability * (currReward + gamma * value_function[currNextState]));

      if (currValue >= maxValue):
        maxValue = currValue
        actionForMaxValue = action

    policy[state] = int(actionForMaxValue)

  return policy


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
    value_function: np.ndarray
      Optimal value function array of length nS
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Tolerance used for stopping criterion based on convergence.
      If the values are changing by less than tol, you should exit.

  Output:
    (np.ndarray, iteration, np.ndarray)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
    policy: Array of integers where each element is the optimal action to take
      from the state corresponding to that index.
  """

  iteration = 0;

  value_function = np.zeros(env.nS)
  value_function.fill(tol)
  policy = np.zeros(env.nS)
  policy.fill(-1);

  while iteration < max_iterations:
    (value_function, currIteration) = policy_evaluation(env, value_function, gamma, (max_iterations - iteration), tol)
    iteration = iteration + currIteration
    (policyStable, policy) = policy_improvement(env, value_function, gamma)
    if policyStable:
      return (value_function, iteration, policy)

  return (value_function, iteration, policy)

def policy_evaluation(env, value_function, gamma, max_iterations=int(1e3), tol=1e-3):

  """
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
    (np.ndarray, iteration, np.ndarray)
    value_function:  Optimal value function
    iteration: number of iterations it took to converge.
  """

  iteration = 0

  for iteration in range(max_iterations):
    delta =  0
    for state in range(env.nS):
      v = value_function[state];
      value_function[state] = 0;
      for action in range(env.nA):
        for nextStateIndex in range(len(env.P[state][action])):
          currP = env.P[state][action][nextStateIndex];
          currProbability = currP[0]
          currNextState = currP[1]
          currReward = currP[2]
          #print("value_function[" + str(state) + "] old = " + str(value_function[state]) + "\n");
          #print("currProbability" + str(currProbability) + ", currReward = " + str(currReward) + ", value_function[currNextState] = " + str(value_function[currNextState]) + "\n");
          # Ints are overflowing, so multiplying the innovation by 0.1
          value_function[state] = value_function[state] + 0.1 * (currProbability * (currReward + gamma * value_function[currNextState]));
          #print("value_function[" + str(state) + "] new = " + str(value_function[state]) + "\n");

      delta = max(delta, abs(v - value_function[state]))

    if delta < tol:
      break

  return (value_function, iteration)

def policy_improvement(env, value_function, gamma):

  """
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
      policyStable: Whether the policy has stabilized.
      policy: Array of integers where each element is the optimal action to take
        from the state corresponding to that index.
  """
  policyStable = True
  for state in range(env.nS):

    action = policy[state]

    v = value_function[state];
    maxValue = -1 * float("inf");
    actionForMaxValue = -1;
    for action in range(env.nA):
      currValue = 0;
      for nextStateIndex in range(len(env.P[state][action])):
        currP = env.P[state][action][nextStateIndex]
        currProbability = currP[0]
        currNextState = currP[1]
        currReward = currP[2]
        currValue += (currProbability * (currReward + gamma * value_function[currNextState]));

      if (currValue >= maxValue):
        maxValue = currValue
        actionForMaxValue = action

    if policy[state] != actionForMaxValue:
      policyStable = False

    policy[state] = int(actionForMaxValue)

    return (policyStable, policy)

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

  # Play around with gamma if you want!
  gamma = 0.9
  # Q3.2.1
  Vs, n_iter = value_iteration(env, gamma)
  policy = policy_from_value_function(env, Vs, gamma)

  (pi_value_function, pi_iteration, pi_policy) = policy_iteration(env, gamma)

  printGridWorld("Values for Value Iteration", Vs, 8, 8, False);
  printGridWorld("Policy (Actions) for Value Iteration", policy, 8, 8, True);

  printGridWorld("Values for Policy Iteration", pi_value_function, 8, 8, False);
  printGridWorld("Policy (Actions) for Policy Iteration", pi_policy, 8, 8, True);
