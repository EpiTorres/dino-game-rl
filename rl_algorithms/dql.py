import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .variables import init_dql
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Class for a Deep Q-Learning neural network
class DQNetwork(torch.nn.Module):
    # Class constructor
    def __init__(self, random_seed, layers, net_path, train):
        super(DQNetwork, self).__init__()
        # Applies a random seed to the network
        torch.manual_seed(random_seed)

        # Stores list of layers
        self.layers = layers

        # Uses the Sequential class of Pytorch to simplify
        # the process of generating a neural net
        self.network = nn.Sequential(*layers)

        # Loads the old weights if currently testing
        if not train:
            print("Loaded saved network weights for testing...")
            self.load_state_dict(torch.load(net_path))
            self.eval()

    # Forward propogation through the neural network
    # Please not this is a required function by Pytorch
    def forward(self, x):
        return self.network(x)


# Class for the Deep Q-Learning agent
class DQLearning():
    # Class constructor
    def __init__(self, game_params):
        # Stores the parameters for the agent
        self.game_params = game_params
        self.dql_params = init_dql()

        # For saving and loading the weights of the network
        self.net_path = os.path.join('rl_algorithms/saved_agents',
                                     self.dql_params['dqnet_file'])

        # Generates and stores the state space and action space
        self.state_space = self.get_state_space()
        self.action_space = self.game_params['move_list']

        # Initializes epsilon for epsilon greedy strategy
        self.epsilon = 1

        # Creates memory for the experience replay
        self.memory = deque(maxlen=self.dql_params['memory_size'])

        # Creates the main neural networks
        self.dqnet_main = DQNetwork(self.dql_params['random_seed'],
                                    self.dql_params['layers'],
                                    self.net_path,
                                    self.game_params['train']).to(DEVICE)
        # Creates the target neurla network by making a deep copy
        self.dqnet_target = copy.deepcopy(self.dqnet_main)

        # Initializes the optimizer for the neural networks
        self.optimizer = optim.Adam(self.dqnet_main.parameters(),
                                    lr=self.dql_params['alpha'])

        # Initialize counter in order to update after a certain amount of time
        self.counter = 0

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

    # Appends the current tuple to memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

    # Takes as input a state tuple. Uses the target neural network to
    # return the action corresponding to the output node with
    # the highest value.
    def get_best_action(self, state):
        # Converts the state tuple to an array
        state_arr = np.asarray(state)
        # Converts the state array to a Pytorch tensor
        state_tensor = torch.tensor(state_arr).float().to(DEVICE)

        main_neural_net = self.dqnet_main
        target_neural_net = self.dqnet_target

        action = 0  # Sets the action to 0 by default so game can run
        # -----------------------------------------------------------------
        # Your Code Begins Here -------------------------------------------

        # (1) Put the main neural network in evaluation mode.

        # (2) With no gradient, use the main neural network and the state
        # tensor to generate output values for each possible action.
        # Don't forget to detach the resulting tensor from the network.

        # (3) Put the main neural network back into training mode.

        # (4) Return the action corresponding to the element in the
        # output tensor with the highest value.

        # Your Code Ends Here ---------------------------------------------
        # -----------------------------------------------------------------

        return action

    # Takes a state tuple as input. Returns an action based on the
    # epsilon greedy strategy.
    def choose_action(self, state):
        epsilon = self.epsilon
        is_training = self.game_params['train']

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

    # Takes as input a minibatch (a list of tuples consisting of a state
    # tuple, action integer, reward integer, next state tuple, and
    # done boolean). Loops through each element of the minibatch to
    # update the main neural network.
    def update_main(self, minibatch):
        # Loops through the experiences in the minibatch.
        for state, action, reward, next_state, done in minibatch:
            # Converts state tuple to array to get state tensor.
            state_arr = np.asarray(state)
            state_tensor = torch.tensor(state_arr).float().to(DEVICE)
            # Converts next state tuple to array to get next state tensor.
            next_state_arr = np.asarray(next_state)
            next_state_tensor = torch.tensor(next_state_arr).float().to(DEVICE)

            main_neural_net = self.dqnet_main
            target_neural_net = self.dqnet_target
            optimizer = self.optimizer

            gamma = self.dql_params['gamma']

            # -----------------------------------------------------------------
            # Your Code Begins Here -------------------------------------------

            # (1) Put the main neural network in training mode. Then, put
            # the target neural network in evaluation mode.

            # (2) Pass the state tensor through the main neural network
            # to get an output tensor. Please note that each value
            # in the tensor corresponds to one possible action.

            # (3) With no gradient, pass the next state tensor
            # into the target neural network to get the output tensor
            # for the next state. Once you get the next state's output
            # tensor, you should be sure to detach it from the network.

            # (4) Calculate the max value from the state's output tensor
            # from step (3).

            # (5) If the value of done is false, 
            # calculate the target reward for the current action using the
            # following equation:
            #       target reward = reward + gamma * max value from step (4)
            # Otherwise, simply set the target reward to be equal to the reward
            # from the current experience in the minibatch.

            # (6) Clone the output tensor from step (2).

            # (7) Use the value from step (5) to replace the current
            # action's value in the cloned tensor from step (6).

            # (8) Detach the cloned tensor.

            # (9) Input the predicted tensor and the target tensor into
            # a loss function to get the loss. For this, all you
            # need to do is uncomment the lines below, replace
            # the predicted_tensor variable with the state output
            # tensor from step (2), and replace the target_tensor
            # variable with the updated cloned tensor from step (8).

            # criterion = torch.nn.MSELoss()
            # loss = criterion(predicted_tensor, target_tensor).to(DEVICE)

            # (10) Use the loss to update the main neural network's weights
            # through backpropagation. For this all you need to do is
            # uncomment the lines below.

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # Your Code Ends Here ---------------------------------------------
            # -----------------------------------------------------------------

    # Updates the target neural networks weights by using a linear
    # interpolation between the main's weights and the target's
    # weights
    def update_target(self):
        tau = self.dql_params['tau']
        for target_param, main_param in zip(self.dqnet_target.parameters(),
                                            self.dqnet_main.parameters()):
            main_node_weight = main_param.data
            target_node_weight = target_param.data

            weight = 0  # Sets the weight to 0 by default so game can run
            # -----------------------------------------------------------------
            # Your Code Begins Here -------------------------------------------

            # (1) Multiply the weight of the main neural network's 
            # current node with tau.

            # (2) Multiply the weight of the target neural network's
            # current node with (1 - tau).

            # (3) Set the value of the weight variable to be equal to the sum
            # of the values from step (1) and step(2).

            # Your Code Ends Here ---------------------------------------------
            # -----------------------------------------------------------------
            target_param.data.copy_(weight)

    # Takes as input a state (tuple), action (integer),
    # dino (Dino class object), and next state (tuple).
    # Adds the given information to the experience
    # replay buffer; if the enough games have passed,
    # uses a random sample from the replay buffer
    # to train the neural networks.
    def update(self, state, action, dino, next_state):
        done = dino.has_crashed

        # Save experience in replay memory
        self.remember(state, action, dino.reward, next_state, done)

        # Learn every update_frequency time steps.
        self.counter = (self.counter + 1) % self.dql_params['update_frequency']
        if self.counter == 0:
            batch_size = self.dql_params['batch_size']
            replay_buffer = self.memory
            # -----------------------------------------------------------------
            # Your Code Begins Here -------------------------------------------

            # (1) If the length of the replay buffer is greater than the
            # batch size:

            #   (a) Create a minibatch by taking a random sample (whose size
            #   is equal to batch size) from the experience replay buffer.

            #   (b) Update the main neural network using the minibatch.

            #   (c) Update the target neural network.

            # Your Code Ends Here ---------------------------------------------
            # -----------------------------------------------------------------

        # Resets the dino's rewards
        dino.reward = 0

    # Saves the weights of the target network
    def save_file(self):
        model_weights = self.dqnet_target.state_dict()
        torch.save(model_weights, self.net_path)
        print("Saved target neural network weights to path:", self.net_path)
