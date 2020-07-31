import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from rl_utils import preparing_the_V_target
from utilfunctions import reshaping_the_histories, scale_state
from rl_utils import calculate_the_A_hat
from rl_utils import cast_A_hat_to_action_form
from rl_utils import preparing_the_V_target

class Agent:
    '''
    the agent class which has the policy.
    - takes actions based on the policy
    - 
    '''
    def __init__(self, nr_actions, gamma=0.99, epsilon=0.02):

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=1)
        optimzer = tf.keras.optimizers.Adam(learning_rate=0.01)
        inputs = keras.layers.Input(shape=(2))
        x = layers.Dense(64, activation='relu', kernel_initializer=initializer)(inputs)
        x = layers.Dense(5, activation='relu', kernel_initializer=initializer)(x)
        output = layers.Dense(nr_actions, activation='softmax', kernel_initializer=initializer)(x)
        output = (output + epsilon) / (1.0 + epsilon * nr_actions)
        self.policy = keras.Model(inputs=inputs, outputs=output)
        self.policy.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=1)
        optimzer = tf.keras.optimizers.Adam(learning_rate=0.01)
        inputs_V = keras.layers.Input(shape=(2))
        x_V = layers.Dense(64, activation='relu', kernel_initializer=initializer)(inputs_V)
        x_V = layers.Dense(5, activation='relu', kernel_initializer=initializer)(x_V)
        output_V = layers.Dense(1, activation='tanh', kernel_initializer=initializer)(x_V)
        self.V = keras.Model(inputs=inputs_V, outputs=output_V)
        self.V.compile(optimizer='adam', loss="mse", metrics=["accuracy"])
        
        self.gamma = gamma
        
    def action_based_on_policy(self, state, env):
        '''
        Returns the chosen action id using the policy for the given state
        '''
        scaled_state = scale_state(state, env)
        probabilities = self.policy.predict(scaled_state)[0]
        #print(state, probabilities)
        nr_actions = len(probabilities)
        chosen_act = np.random.choice(nr_actions, p=probabilities)
        return chosen_act

    def learning(self, histories, env):
        '''
        the learning happens here:
        1- first the value function is learned,
        2- the policy is learned.
        '''
        print("...    reshaping the data")
        tmp_histories = reshaping_the_histories(histories)

        # 1.1 preparing the target value for traingin V
        print("...    preparing target for V-calculation")
        target_for_V_training = preparing_the_V_target(self, tmp_histories, env)

        # 1.2 fitting the V
        self.V.fit(x=tmp_histories.scaled_state_history, y=target_for_V_training, epochs=1, verbose=0)

        print("...    training the value-function")

        # 2.1 calculating the advantages
        A_hat = calculate_the_A_hat(self, tmp_histories, env)
        A_hat = cast_A_hat_to_action_form(A_hat, tmp_histories)
        
        # 2.2 fitting the policy
        print("...    training the policy")
        print("...        with sample size of ", np.shape(tmp_histories.scaled_state_history)[0])
        
        fitting_log = self.policy.fit(x=tmp_histories.scaled_state_history, y=A_hat, epochs=1, verbose=0)
