import random
import argparse
import pygame
from game_classes.dino import Dino
from game_classes.obstacle import Obstacle
from game_classes.background import Ground, Scoreboard, Cloud
from rl_algorithms.variables import init_game
from rl_algorithms.ql import QLearning
from rl_algorithms.dql import DQLearning


class DinoGame():
    # Class constructor
    def __init__(self, game_params):
        self.game_params = game_params

        self.state = (0, 0, 0, 0, 0)
        self.action = 0
        self.next_state = (0, 0, 0, 0, 0)
        self.episodes = game_params['episodes']

        if game_params['agent_alg'] == "ql":
            self.agent = QLearning(game_params)
        elif game_params['agent_alg'] == "dql":
            self.agent = DQLearning(game_params)
        else:
            self.agent = None

        self.game_count = 1
        self.total_score = 0
        self.high_score = 0

        self.game_speed = game_params['start_speed']
        self.speed_counter = 0

        self.obstacles = pygame.sprite.Group()
        Obstacle.containers = self.obstacles
        self.last_obs = None
        self.dino = Dino(game_params)
        self.ground = Ground(game_params)
        self.scoreboard = Scoreboard(game_params)
        self.clouds = pygame.sprite.Group()
        Cloud.containers = self.clouds

        self.game_running = True

    # Updates all of the sprites on the screen
    def update_game_screen(self):
        self.scoreboard.update(self.dino.score)
        # Creates a new cloud if able
        if len(self.clouds) < 5 and random.randrange(0, 30) == 1:
            Cloud(self.game_params, self.game_speed)

        if pygame.display.get_surface() is not None:
            screen = self.game_params['screen']
            screen.fill(self.game_params['background_col'])
            self.ground.draw(screen)
            self.clouds.draw(screen)
            self.scoreboard.draw(screen)
            self.obstacles.draw(screen)
            self.dino.draw(screen)
            pygame.display.update()
            self.game_params['clock'].tick(self.game_params['FPS'])

    # Updates the positions of the dino, the obstacles,
    # and other game objects. Updates the game speed
    # if appropriate.
    def update_game_state(self):
        self.obstacles.update(self.game_speed)
        self.dino.update(self.game_speed)
        if self.game_params['display']:
            self.ground.update(self.game_speed)
            self.clouds.update()
            self.scoreboard.update(self.dino.score)

        # Updates the game speed
        if (self.speed_counter == 699
           and self.game_speed < self.game_params['max_speed']):
            self.game_speed += 1
        self.speed_counter = (self.speed_counter + 1) % 700

    # If the dino has crashed into any obstacles, gives the
    # dino a penalty. If the dino is currently avoiding an
    # obstacle, gives the dino a reward. If the dino
    # has just passed an obstacle, updates the dino's score.
    def check_collisions(self):
        for obs in self.obstacles:
            if self.dino.crashed(obs):
                self.dino.give(self.game_params['dino_penalty'])
            elif self.dino.is_avoiding(obs):
                self.dino.give(self.game_params['dino_reward'])
            elif self.dino.just_avoided(obs):
                self.dino.update_score()

    # Creates a new obstacle if there are none on the screen or if
    # the last obstacle is far enough from the right of the screen.
    def create_obstacle(self):
        if len(self.obstacles) == 0:
            obs = Obstacle(self.game_params, self.game_speed)
            self.last_obs = obs
        else:
            dist = self.game_params['scr_width'] - self.last_obs.rect.right
            rand = random.randrange(0, 10)
            if (dist >= self.last_obs.max_gap
               or (dist > self.last_obs.min_gap and rand == 0)):
                obs = Obstacle(self.game_params, self.game_speed)
                self.last_obs = obs

    # Updates and resets all of the values necessary in order
    # to start a fresh run. Ends the game if the game count
    # is greater than or equal to the max number of episodes.
    def start_new_run(self):
        self.total_score += self.dino.score  # Updates total score
        # Updates high score if appropriate
        if (self.dino.score > self.high_score):
            self.high_score = self.dino.score

        # Decreases epsilon for epsilon-greedy strategy
        if self.agent.epsilon > 0:
            new_epsilon = 1 - self.game_count*self.game_params['epsilon_decay']
            self.agent.epsilon = new_epsilon

        # Ends the game when the game count exceeds the number of episodes
        if self.game_count >= self.episodes:
            self.end_game()

        # Resets the game state
        self.game_count += 1
        self.game_speed = self.game_params['start_speed']
        self.dino.reset(self.game_speed)
        self.clouds.empty()
        self.obstacles.empty()
        self.last_obs = None

    # Prints the average score and high score for all episodes
    # of this game. If training, saves agent as a file.
    def end_game(self):
        self.game_running = False

        self.total_score += self.dino.score
        avg_score = self.total_score/self.game_count
        print("Average Score : " + str(avg_score))

        if (self.dino.score > self.high_score):
            self.high_score = self.dino.score
        print("High Score : " + str(self.high_score))

        if self.game_params['train']:
            self.agent.save_file()

    # Runs the main game loop.
    def run(self):
        while self.game_running:
            # Quits the game when the Pygame popup is closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_game()
                    pygame.quit()
                    quit()

            # Performs an action if the dino is not jumping
            if not self.dino.is_jumping:
                self.state = self.agent.get_state(self.dino,
                                                  self.obstacles)
                self.action = self.agent.choose_action(self.state)

                if self.action == 0:
                    self.dino.run()
                elif self.action == 1:
                    self.dino.jump()
                else:
                    self.dino.duck()

                self.update_game_state()
                self.next_state = self.agent.get_state(self.dino,
                                                       self.obstacles)
            # Otherwise, simply updates the game state
            else:
                self.update_game_state()

            # Checks if the dinosaur has crashed with any obstacles or
            # is avoiding any obstacles in order to set its
            # reward/punishment.
            self.check_collisions()

            # Updates the agent's strategy if training
            if self.game_params['train']:
                # Only updates the strategy if the dino has crashed
                # or if the dino is not jumping
                if self.dino.has_crashed or not self.dino.is_jumping:
                    self.agent.update(self.state,
                                      self.action,
                                      self.dino,
                                      self.next_state)

            # If possible, creates a new obstacle.
            self.create_obstacle()

            # If appropriate, updates the game's screen.
            if self.game_params['display']:
                self.update_game_screen()

            # Starts a new run if the dino has crashed or if
            # if the score of the dino has exceeded the
            # training cutoff score.
            if self.dino.has_crashed or (self.game_params['train']
               and self.dino.score > self.game_params['training_cutoff']):
                self.start_new_run()


######################################
if __name__ == "__main__":
    # Set options to activate or deactivate the game view, and its speed
    game_params = init_game()
    parser = argparse.ArgumentParser()

    ##################################################################
    # Trains the agent if True. Default is False.
    parser.add_argument("--train",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    # Tests the agent if True. Default is False.
    parser.add_argument("--test",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    ##################################################################
    # Displays the episodes when training if True. Default is False.
    parser.add_argument("--train-display",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    # Displays the episodes when testing if True. Default is True.
    parser.add_argument("--test-display",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    ##############################################################
    # Number of episodes used to train the agent if --train is used.
    # Default is 1000.
    parser.add_argument("--train-episodes", nargs='?', type=int, default=1000)

    # Number of episodes used to test the agent if --test is used.
    # Default is 100.
    parser.add_argument("--test-episodes", nargs='?', type=int, default=100)

    ##################################################################
    # Runs Q-Learning if True
    # Default is False.
    parser.add_argument("--ql",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    # Runs Deep Q-Learning if True
    # Default is False.
    parser.add_argument("--dql",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    args = parser.parse_args()
    print("Command-line args:", args)

    if args.train:
        game_params['train'] = True
        game_params['display'] = args.train_display
        game_params['episodes'] = args.train_episodes
        game_params['epsilon_decay'] = 1 / args.train_episodes
        if args.ql:
            print("\nTraining Q-Learning...")
            game_params['agent_alg'] = "ql"
            train_game = DinoGame(game_params)
            train_game.run()
        if args.dql:
            print("\nTraining Deep Q-Learning...")
            game_params['agent_alg'] = "dql"
            train_game = DinoGame(game_params)
            train_game.run()

    if args.test:
        game_params['train'] = False
        game_params['display'] = args.test_display
        game_params['episodes'] = args.test_episodes
        game_params['epsilon_decay'] = 1 / args.test_episodes
        if args.ql:
            print("\nTesting Q-Learning for ", args.test_episodes,
                  "episodes...")
            game_params['agent_alg'] = "ql"
            test_game = DinoGame(game_params)

            test_game.run()
        if args.dql:
            print("\nTesting Deep Q-Learning for ", args.test_episodes,
                  "episodes...")
            game_params['agent_alg'] = "dql"
            test_game = DinoGame(game_params)
            test_game.run()
