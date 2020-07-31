import numpy as np
import random
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
from rl_utils import monitoring_performance
from utilfunctions import find_initial_state
from utilfunctions import initializer
from utilfunctions import one_hot
from utilfunctions import update_state_step
from utilfunctions import scale_state


import os
# the size of the system
SYSTEM_SIZE = 8

# creating the environment
env = Environment(SYSTEM_SIZE)

# creating the agent with 4 actions
# in case you want to change this it should be also changed in the Environment class
nr_actions = 4
agent_1 = Agent(nr_actions=nr_actions, gamma=0.95)
agent_2 = Agent(nr_actions=nr_actions, gamma=0.95)
agent_1.policy = load_model('./training-results/policy-not-trained-agent-system-size-'+str(SYSTEM_SIZE))
agent_2.policy = load_model('./training-results/policy-trained-agent-system-size-'+str(SYSTEM_SIZE))



# setting the random seeds
random.seed(1)
np.random.seed(2)

plt.close('all')
plt.ion()
agent_1.V = load_model('./training-results/v-trained-agent-system-size-'+str(SYSTEM_SIZE))
heat = [agent_1.V.predict(scale_state(np.array([[x, y]]), env)) for x in range(env.SYSTEM_SIZE) for y in range(env.SYSTEM_SIZE)]
heat = np.reshape(np.array(heat), (SYSTEM_SIZE,SYSTEM_SIZE))
epsilon = 0.00001
shifted_heat = heat - np.min(heat) + epsilon
log_heat = np.log(shifted_heat)
plt.imshow(log_heat)
plt.show()

n = SYSTEM_SIZE
plt.scatter(n-1, n-1, s=100, c='orange', marker='s')
plt.axis([-1, n, -1, n])
plt.axes().set_aspect('equal')
plt.scatter(0, 0, c='red')


initial_state = np.array([[0, 0]])

print("...    initial state is "+str(initial_state))

state_1, terminated_1, steps_1 = initializer(initial_state[0, 0], initial_state[0, 1])
state_2, terminated_2, steps_2 = initializer(initial_state[0, 0], initial_state[0, 1])

step = 0
filename = ''
animations_dir = 'animations/'
os.makedirs(animations_dir)

while not terminated_1 and not terminated_2:

    # the first agent
    # print("agent 1")
    action_id = agent_1.action_based_on_policy(state_1, env)
    one_hot_action = one_hot(action_id, nr_actions)
    new_state, reward, terminated_1 = env.step(action_id, state_1)
    scaled_state_1 = scale_state(state_1, env)
    #histories_1.appending(reward, scaled_state_1, one_hot_action)
    plt.scatter(state_1[0, 0], state_1[0, 1], s=100, c='#C1C7C9', marker='s')
    plt.scatter(new_state[0, 0], new_state[0, 1], s=50, c='red')
    plt.show()
    plt.pause(0.1)

    state_1, steps_1 = update_state_step(new_state, steps_1)

    # the second agent
    # print("agent 2")
    action_id = agent_2.action_based_on_policy(state_2, env)
    one_hot_action = one_hot(action_id, nr_actions)
    new_state, reward, terminated_2 = env.step(action_id, state_2)
    scaled_state_2 = scale_state(state_2, env)
    #histories_2.appending(reward, scaled_state_2, one_hot_action)
    plt.scatter(state_2[0, 0], state_2[0, 1], s=100, c='#C1C7C9', marker='s')
    plt.scatter(new_state[0, 0], new_state[0, 1], s=50, c='blue')
    plt.show()
    plt.pause(0.1)

    state_2, steps_2 = update_state_step(new_state, steps_2)

    if (step < 10):
        filename = 'state_00'+str(step)+'.png'
    elif (step < 100):
        filename = 'state_0'+str(step)+'.png'
    elif (step < 1000):
        filename = 'state_'+str(step)+'.png'
    plt.savefig(animations_dir+filename)
    step += 1
    
    print("...    terminated at: "+str(steps_1))
