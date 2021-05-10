import pygame
import torch.nn as nn


# Manually defines general parameters for gameplay
def init_game():
    ##########################
    # Initializes pygame & display
    pygame.init()
    params = dict()
    params['scr_width'] = 600
    params['scr_height'] = 200
    params['scr_size'] = (params['scr_width'], params['scr_height'])
    params['screen'] = pygame.display.set_mode(params['scr_size'])
    pygame.display.set_caption("Dino Game")
    params['bottom_padding'] = 10
    params['ground_pos'] = params['scr_height'] - params['bottom_padding']
    params['start_speed'] = 6
    params['max_speed'] = 13
    params['black'] = (0, 0, 0)
    params['white'] = (255, 255, 255)
    params['background_col'] = (235, 235, 235)
    params['clock'] = pygame.time.Clock()
    params['FPS'] = 60

    ##########################
    # For the Dino class
    params['dino_width'] = 44
    params['dino_height'] = 57
    params['dino_duck_width'] = 59
    params['dino_duck_height'] = 25
    params['dino_img'] = 'dino.png'
    params['dino_duck_img'] = 'dino_ducking.png'

    params['initial_jump_velocity'] = 12
    params['gravity'] = 0.6
    params['move_list'] = [0, 1, 2]  # running, jumping, ducking

    params['dino_penalty'] = -10
    params['dino_reward'] = 5

    ##########################
    # For the Obstacles class
    params['min_gap_coeff'] = 0.6
    params['max_gap_coeff'] = 1.5
    params['max_cacti_length'] = 3
    # For small cacti obstacles
    params['sm_cacti_width'] = 17
    params['sm_cacti_height'] = 35
    params['sm_cacti_gap'] = 120
    params['sm_cacti_img'] = 'sm_cacti.png'
    # For large cacti obstacles
    params['lg_cacti_width'] = 25
    params['lg_cacti_height'] = 50
    params['lg_cacti_gap'] = 120
    params['lg_cacti_img'] = 'lg_cacti.png'
    # For pterodactyl obstacles
    params['pter_width'] = 46
    params['pter_height'] = 40
    params['pter_y_pos'] = [150 + params['pter_height']/2,
                            125 + params['pter_height']/2,
                            100 + params['pter_height']/2]
    params['pter_gap'] = 150
    params['pter_img'] = 'pter.png'

    ##########################
    # For background classes
    params['cloud_width'] = 64  # Cloud class
    params['cloud_height'] = 30
    params['cloud_img'] = 'cloud.png'

    params['ground_img'] = 'ground.png'  # Ground class

    params['numbers_img'] = 'numbers.png'  # Scoreboard class

    ##########################
    # Params that are adjusted via command-line arguments
    params['display'] = False
    params['episodes'] = 2500
    params['epsilon_decay'] = 1/2500  # How much epsilon decreases per game
    params['train'] = True  # True if the agent is currently being trained
    params['agent_alg'] = "ql"  # Can be either "ql" or "dql"

    ##########################
    # Misc. params
    params['training_cutoff'] = 500  # max score during training

    return params


# Manually defines the parameters for the Q-Learning algorithm
def init_ql():
    ql_params = dict()

    # Hyperparameters for Q-table
    ql_params['alpha'] = 0.75  # Learning rate
    ql_params['gamma'] = 0.5  # Discount term

    # Misc. params
    ql_params['qtable_file'] = 'qtable.csv'  # File with saved qtable

    return ql_params


# Manually defines the parameters for the Deep Q-Learning algorithm
def init_dql():
    dql_params = dict()

    # Hyperparameters for DQNetwork class
    dql_params['tau'] = 0.0005  # parameter for the soft update
    dql_params['alpha'] = 0.0005  # Learning rate
    dql_params['gamma'] = 0.5  # Discount term

    # Defines the layers to be used by the DQNetwork class
    # See the following for more explanation:
    #   https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
    dql_params['layers'] = [nn.Linear(5, 25), nn.ReLU(),  # 1st hidden layer
                            nn.Linear(25, 25), nn.ReLU(),  # 2nd hidden layer
                            nn.Linear(25, 25), nn.ReLU(),  # 3rd hidden layer
                            nn.Linear(25, 3)]   # Output layer

    # Params for experience replay buffer
    dql_params['memory_size'] = 10000  # Replay buffer size
    dql_params['batch_size'] = 100  # Minibatch size
    dql_params['update_frequency'] = 5  # Number of steps between learning

    # Misc. params
    dql_params['random_seed'] = 22
    dql_params['dqnet_file'] = 'dqn.h5'  # File with saved network weights

    return dql_params
