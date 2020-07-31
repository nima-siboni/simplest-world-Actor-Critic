import numpy as np
from utilfunctions import scale_state

def calculate_the_A_hat(agent, histories, env):
    '''
    calculating the advantageto go hat{A} for each (state, action) in the history.
    
    Key arguments:
    histories -- the rewards for all the visited states
    gamma -- the discount factor

    output:
    the hat{A} which is of the same shape as reward_history.
    '''

    # prepare the s_{t+1}
    s = histories.scaled_state_history
    s_prime = np.roll(s, -1, axis=0)
    s_prime[-1, :] = scale_state(env.TERMINAL_STATE, env)
    
    tmp = agent.gamma * agent.V.predict(s_prime) - agent.V.predict(s)
    tmp = tmp.reshape(-1, 1)
    A_hat = histories.reward_history + tmp
    
    return A_hat

def cast_A_hat_to_action_form(A_hat, histories):

    output = np.multiply(histories.action_history, A_hat)

    return output


def monitoring_performance(log, training_id, steps, initial_state, env, write_to_disk=True):
    '''
    returns a log (a numpy array) which has some analysis of the each round of training.

    Key arguments:

    training_id -- the id of the iteration which is just finished.
    steps -- the number of steps for the agent to reach to the terminal state in this iteration
    initial_state -- the initial state of the agent in this iteration
    env -- the environment (only the terminal_state is extracted from it)
    write_to_disk -- a flag for writting the performance to the disk

    Output:

    a numpy array with info about the iterations and the learning
    '''

    steps_for_the_optimal_policy = np.sum(env.TERMINAL_STATE - initial_state)

    assert steps_for_the_optimal_policy > 0

    performance = steps_for_the_optimal_policy / steps

    if training_id == 0:
        log = np.array([[training_id, performance, steps]])
    else:
        log = np.append(log, np.array([[training_id, performance, steps]]), axis=0)

    if write_to_disk:
        np.savetxt('reward_vs_iteration.dat', log)

    return log


class Histories():
    '''
    just a class to hold data
    '''
    def __init__(self):
        self.scaled_state_history = []
        self.reward_history = []
        self.action_history = []

    def appending(self, reward, scaled_state, one_hot_action):
        self.reward_history.append(reward)
        self.scaled_state_history.append(scaled_state)
        self.action_history.append(one_hot_action)


def preparing_the_V_target(agent, histories, env):
    '''
    Calculting the y as
    y_{t} = r(a_{t}, s_{t}) + gamma * V(s_{t+1})
    '''
    # create the y array with zeros
    # y = np.zeros_like(histories.reward_history)

    # y_{t} = r(a_{t}, s_{t})
    y = histories.reward_history + 0.0

    # prepare the s_{t+1}
    next_scaled_state = np.roll(histories.scaled_state_history, -1, axis=0)
    next_scaled_state[-1, :] = scale_state(env.TERMINAL_STATE, env)

    # calculate V(s_{t+1})
    target = y + agent.gamma * agent.V.predict(next_scaled_state)

    return target
