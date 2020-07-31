import numpy as np
import random
from agent import Agent
from environment import Environment
from utilfunctions import find_initial_state
from utilfunctions import initializer
from utilfunctions import one_hot
from utilfunctions import scale_state
from utilfunctions import update_state_step
from rl_utils import Histories
from rl_utils import monitoring_performance

# the size of the system
SYSTEM_SIZE = 8

# creating the environment
env = Environment(SYSTEM_SIZE)

# creating the agent with 4 actions
# in case you want to change this it should be also changed in the Environment class
nr_actions = 4
agent = Agent(nr_actions=nr_actions, gamma=0.99, epsilon=0.02)

# number of trainings
ROUNDS_OF_TRAINING = SYSTEM_SIZE * SYSTEM_SIZE * 500


# setting the random seeds
random.seed(1)
np.random.seed(3)
training_log = np.array([])
histories = Histories()

expr_per_learn = 1 #env.SYSTEM_SIZE * env.SYSTEM_SIZE * nr_actions

agent.policy.save('./training-results/policy-not-trained-agent-system-size-'+str(SYSTEM_SIZE))
agent.V.save('./training-results/v-not-trained-agent-system-size-'+str(SYSTEM_SIZE))



for training_id in range(ROUNDS_OF_TRAINING):

    print("\nround: "+str(training_id))

    # finding a initial state which is not the terminal_state
    initial_state = find_initial_state(env)
    #initial_state = np.array([[0, 0]])
    print("...    initial state is "+str(initial_state))

    state, terminated, steps = initializer(initial_state[0, 0], initial_state[0, 1])

    while not terminated:

        action_id = agent.action_based_on_policy(state, env)
        one_hot_action = one_hot(action_id, nr_actions)
        
        new_state, reward, terminated = env.step(action_id, state)

        scaled_state = scale_state(state, env)
        
        histories.appending(reward, scaled_state, one_hot_action)

        #print("state", state, "scla", scaled_state, "one_hot_action", one_hot_action, "new_state", new_state, "reward", reward)
        state, steps = update_state_step(new_state, steps)
        
        if steps % 100 == 0:
            print("         step "+str(steps)+"  state is "+str(state))

    print("...    the terminal_state is reached after "+str(steps)+" instead of "+ str(np.sum(env.TERMINAL_STATE-initial_state))+" steps")
    if (training_id % expr_per_learn) == (expr_per_learn - 1):
        agent.learning(histories, env)
        histories = Histories()
        print("...    saving the policy")
        agent.policy.save('./training-results/policy-trained-agent-system-size-'+str(SYSTEM_SIZE), save_format='tf')
        print("...    saving the value function")
        agent.V.save('./training-results/v-trained-agent-system-size-'+str(SYSTEM_SIZE), save_format='tf')

    training_log = monitoring_performance(training_log, training_id, steps, initial_state, env, write_to_disk=True)

    print("round: "+str(training_id)+" is finished.")
