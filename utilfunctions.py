import numpy as np


def one_hot(chosen_act, nr_actions):
    '''
    turn the chosen action to a one hotted vector
    '''
    tmp = np.zeros((1, nr_actions))
    tmp[0, chosen_act] = 1
    return tmp


def initializer(x0=0, y0=0):
    '''
    returns:
    - the initial state, 
    - a bunch of empty arrays,
    - set the terminated to False
    - reset the steps to 0
    '''

    state = np.array([[x0, y0]])
    terminated = False
    steps = 0
    return state, terminated, steps


def shape_adopter(history, m):
    '''
    convert a (x, 1, y)
    '''
    history = np.array(history)
#    import pdb; pdb.set_trace()
#    _, _, m = np.shape(history)
    history = history.reshape(-1, m)

    return history


def reshaping_the_histories(histories):
    '''
    Capsulating the shape_adopter
    '''
    scaled_state_history = shape_adopter(histories.scaled_state_history, 2)
    action_history = shape_adopter(histories.action_history, 4)
    reward_history = shape_adopter(histories.reward_history, 1)

    histories.scaled_state_history = scaled_state_history
    histories.action_history = action_history
    histories.reward_history = reward_history

    return histories


def find_initial_state(env):
    '''
    Finding an initial state which is not the final state
    '''
    initial_state = env.TERMINAL_STATE + 0
    while np.array_equal(initial_state, env.TERMINAL_STATE):
        initial_state = np.random.randint(low=0, high=env.SYSTEM_SIZE, size=(1, 2))

    return initial_state


def update_state_step(new_state, step):
    state = new_state + 0
    step = step + 1
    return state, step


def scale_state(state, env):
    '''
    scaling the coordinates between -1.0 and 1.0
    returns the scaled state
    '''

    scaled_state = (2.0 * state) / (env.SYSTEM_SIZE - 1.0) - 1.0

    return scaled_state
