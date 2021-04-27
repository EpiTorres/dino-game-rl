import os
import random
import numpy as np
from .variables import init_ql


# Class for the Q-Learning agent
class QLearning():
    # Class constructor
    def __init__(self, game_params):
        self.game_params = game_params
        self.ql_params = init_ql()

        # For saving and loading the weights of the network
        self.qtable_path = os.path.join('rl_algorithms/saved_agents',
                                        self.ql_params['qtable_file'])

        self.state_space = self.get_state_space()
        self.action_space = self.game_params['move_list']

        self.epsilon = 1
        self.qtable = self.load_qtable()

    # Returns Q-Table with all values initialized to 0 if training.
    # Otherwise, loads the Q-table saved in the csv file.
    def load_qtable(self):
        if self.game_params['train']:
            state_size = len(self.state_space)
            action_size = len(self.action_space)
            return np.zeros([state_size, action_size])
        else:
            print("Loaded saved Q-Table for testing...")
            return np.loadtxt(self.qtable_path, delimiter=',')

    # Uses the starting speed, max speed, and info about each type of obstacle
    # to set up the state space.
    def get_state_space(self):
        state_space = []
        start_speed = self.game_params['start_speed']
        max_speed = self.game_params['max_speed']

        # Appends an empty state for whenever no obstacles are on screen
        state_space.append((0, 0, 0, 0, 0))

        # Loops through all possible game speeds
        for s in range(start_speed, max_speed+1):
            # Loops through possible x-values for the obstacles
            for x in range(0, self.game_params['scr_width'] + 1):
                # Appends tuples for Pterodactyl obstacles
                for y in range(0, len(self.game_params['pter_y_pos'])):
                    pter_tuple = (x,
                                  self.game_params['pter_y_pos'][y],
                                  self.game_params['pter_width'],
                                  self.game_params['pter_height'],
                                  s)  # speed
                    state_space.append(pter_tuple)

                # Appends tuples for Cacti obstacles
                max_cacti_length = self.game_params['max_cacti_length']
                for n in range(1, max_cacti_length + 1):
                    # For small cacti
                    sm_cacti_tuple = (x,
                                      self.game_params['ground_pos'],
                                      n * self.game_params['sm_cacti_width'],
                                      self.game_params['sm_cacti_height'],
                                      s)
                    state_space.append(sm_cacti_tuple)

                    # For large cacti
                    lg_cacti_tuple = (x,
                                      self.game_params['ground_pos'],
                                      n * self.game_params['lg_cacti_width'],
                                      self.game_params['lg_cacti_height'],
                                      s)
                    state_space.append(lg_cacti_tuple)

        return state_space

    # Takes as input dino (a Dino class object) and obstacles (a list
    # of Obstacle class objects). If obstacles is not empty, returns a
    # tuple with the left position, bottom position, width, height, and
    # game_speed of the nearest obstacle to dino. Otherwise, returns a
    # tuple with zeros.
    def get_state(self, dino, obstacles):
        state = (0, 0, 0, 0, 0)
        # Loops through the obstacles to get the closest one
        for obs in obstacles:
            if obs.rect.right > dino.rect.left and obs.rect.left >= 0:
                state = (obs.rect.left,
                         obs.rect.bottom,
                         obs.rect.width,
                         obs.rect.height,
                         obs.speed)
                break
        return state

    # Returns a random action from the action space.
    def get_random_action(self):
        action_space = self.action_space

        action = 0  # Sets the action to 0 by default so game can run
        # -----------------------------------------------------------------
        # Your Code Begins Here -------------------------------------------

        # (1) Return a random value from the action space.
        # Please note that the action space is a list.

        # Your Code Ends Here ---------------------------------------------
        # -----------------------------------------------------------------

        return action

    # Returns the action with the highest Q-value for the given state.
    def get_best_action(self, state):
        state_space = self.state_space
        qtable = self.qtable

        action = 0  # Sets the action to 0 by default so game can run
        # -----------------------------------------------------------------
        # Your Code Begins Here -------------------------------------------

        # (1) Get the index of the given state in the state space. 
        # Please keep in mind that the state space is a list.

        # (2) Use the state's index and the Q-Table to get the
        # Q-Values for each action.

        # (3) Return the action corresponding to the highest Q-Value.

        # Your Code Ends Here ---------------------------------------------
        # -----------------------------------------------------------------

        return action

    # Takes a state tuple as input. Returns an action based on the
    # epsilon greedy strategy.
    def choose_action(self, state):
        epsilon = self.epsilon
        is_training = self.game_params['train']  # True if agent is training

        action = 0  # Sets the action to 0 by default so game can run
        # -----------------------------------------------------------------
        # Your Code Begins Here -------------------------------------------

        # (1) Return a random action if a random value from [0, 1)
        # is less than epsilon and the agent is training.

        # (2) Otherwise, return the best expected action for
        # the given state.

        # Your Code Ends Here ---------------------------------------------
        # -----------------------------------------------------------------

        return action

    # Takes as input a state (tuple), action (integer),
    # dino (Dino class object), and next state (tuple).
    # Updates the Q table (using Bellman equation)
    # based on the state, action, next state
    def update(self, state, action, dino, next_state):
        reward = dino.reward
        alpha = self.ql_params['alpha']
        gamma = self.ql_params['gamma']

        state_space = self.state_space
        qtable = self.qtable

        # -----------------------------------------------------------------
        # Your Code Begins Here -------------------------------------------

        # (1) Use the state space to get the index of the current
        # state in the Q-Table

        # (2) Get the current Q-Value in the Q-Table for the current state
        # and action.

        # (3) Use the state space to get the index of the next
        # state in the Q-Table

        # (4) Get the highest Q-Value in the Q-Table for the next state

        # (5) Calculate the new Q-Value using the Bellman equation.

        # (6) Replace the old Q-Value for the current action with the new
        # Q-value.

        # Your Code Ends Here ---------------------------------------------
        # -----------------------------------------------------------------

        # Resets the dino's reward
        dino.reward = 0  # DO NOT REMOVE

    # Saves the current Q-table as a csv file
    def save_file(self):
        np.savetxt(self.qtable_path, self.qtable, delimiter=",")
        print("Saved Q-Table to path:", self.qtable_path)
