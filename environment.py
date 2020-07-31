import numpy as np


class Environment:
    '''
    A class for the environment:

    - creates a lattice model of the given size
    - the list of possible allowed actions are also set in the initializer.
    - the terminal state is assigned here,
    - whether the agent has reached the terminal state is also stored here(!!!)
    - reward for different events are also set here
        -- for each step
        -- for reaching the final state
        -- for hitting the wall

    The step method returns:
    - the new state, and
    - the reward,
    for the given input action id and for the given state.
    '''

    def __init__(self, system_size=5):

        self.actions_dict = {
            0: [1, 0],
            1: [0, 1],
            2: [-1, 0],
            3: [0, -1]
        }

        self.SYSTEM_SIZE = system_size
        self.TERMINAL_STATE = np.array([[system_size - 1, system_size - 1]])
        self.NR_ACTIONS = 4
        self.reward_for_each_step = -1
        self.reward_for_final_state = 10
        self.reward_for_hitting_the_wall = -4

    def step(self, action_id, state):
        '''
        given the action_id, and the state it takes the step and returns:
        - the new state, and
        - the reward.
        side effect: changes the value of terminated variable if the 
        agent reaches the terminal state.

        Key arguments:
        action_id -- the number associated with the action.
        state -- the current state
        '''

        terminated = False

        chosen_action = self.actions_dict.get(action_id)

        new_state = state + np.array(chosen_action, dtype=int)

        rejected = (new_state < 0).any() or (new_state >= self.SYSTEM_SIZE).any()

        if rejected:
            new_state = state + 0
            reward = self.reward_for_hitting_the_wall
        else:
            if (np.array_equal(new_state, self.TERMINAL_STATE)):
                #print("TERMINAL_STATE is reached")
                reward = self.reward_for_final_state
                terminated = True
            else:
                reward = self.reward_for_each_step

        return new_state, reward, terminated
