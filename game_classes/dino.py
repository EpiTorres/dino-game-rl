import os
import pygame


# Class that handles game logic for agent/player of the game
class Dino():
    def __init__(self, game_params):
        # Stores width and height of dino
        self.width = game_params['dino_width']
        self.height = game_params['dino_height']

        # Stores width and height of dino when ducking
        self.duck_width = game_params['dino_duck_width']
        self.duck_height = game_params['dino_duck_height']

        # Gets the sprites and rect of the dino
        self.sprites = self.load_sprites(game_params['dino_img'],
                                         5,
                                         self.width,
                                         self.height)
        self.rect = self.sprites[0].get_rect()
        self.duck_sprites = self.load_sprites(game_params['dino_duck_img'],
                                              2,
                                              self.duck_width,
                                              self.height)
        self.image = self.sprites[0]
        self.sprite_idx = 0
        self.counter = 0

        # Sets the dino's position
        self.ground_pos = game_params['ground_pos']
        self.rect.bottom = self.ground_pos
        self.rect.left = int(game_params['scr_width']/15)

        # For movement and jumping
        self.movement = [0, 0]
        self.jump_speed = game_params['initial_jump_velocity']
        self.gravity = game_params['gravity']
        self.game_speed = game_params['start_speed']

        # Stores the status of the dino
        self.is_ducking = False
        self.is_jumping = False
        self.has_crashed = False

        # Store's the dino's score and reward
        self.score = 0
        self.reward = 0

    # Returns a list of images corresponding to the
    # dino's sprites.
    def load_sprites(self, sheetname, sprite_num,
                     sprite_width, sprite_height):
        # Loads the sprite sheet
        path = os.path.join('game_classes/sprites', sheetname)
        sheet = pygame.image.load(path).convert()
        sheet_rect = sheet.get_rect()

        sprites = []

        # Gets the original dimensions for each sprite
        size_x = sheet_rect.width/sprite_num
        size_y = sheet_rect.height

        # Loops through all sprites in the sprite sheet
        # and appends them to the sprites list
        for i in range(sprite_num):
            rect = pygame.Rect((i*size_x, 0, size_x, size_y))

            image = pygame.Surface(rect.size).convert()
            image.blit(sheet, (0, 0), rect)

            colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)

            sprite_dims = (sprite_width, sprite_height)
            image = pygame.transform.scale(image, sprite_dims)

            sprites.append(image)

        return sprites

    # Updates the sprites based on the dino's status
    def update_sprites(self):
        if self.is_ducking:
            if self.counter == 0:
                self.sprite_idx = (self.sprite_idx + 1) % 2
                self.image = self.duck_sprites[self.sprite_idx]
        else:
            if self.is_jumping:
                self.sprite_idx = 0
            elif self.counter == 0:
                self.sprite_idx = (self.sprite_idx + 1) % 2 + 2
            self.image = self.sprites[self.sprite_idx]

    # Draws the dino on screen
    def draw(self, screen):
        screen.blit(self.image, self.rect)

    # Dino becomes upright when running
    def run(self):
        self.rect.width = self.width
        self.rect.height = self.height
        self.is_ducking = False

    # Resets dino's status and position if dino is close enough to ground
    def check_bounds(self):
        if self.rect.bottom >= self.ground_pos:
            self.rect.bottom = self.ground_pos
            self.is_jumping = False
            self.movement = [0, 0]

    # Initiates a jump
    def jump(self):
        self.rect.width = self.width
        self.rect.height = self.height
        self.movement[1] = -self.jump_speed + self.game_speed/10
        self.is_jumping = True
        self.is_ducking = False

    # Dino begins ducking
    def duck(self):
        self.rect.width = self.duck_width
        self.rect.height = self.duck_height
        self.is_ducking = True

    # Returns true if the dino has crashed or is crashing into
    # an obstacle. Returns false otherwise.
    def crashed(self, obs):
        if not self.has_crashed:
            self.has_crashed = pygame.sprite.collide_mask(self, obs)
        return self.has_crashed

    # Returns True whenever the dino just avoided the obstacle obs;
    # Returns False otherwise.
    def just_avoided(self, obs):
        if (self.rect.left > obs.rect.right and not obs.avoided):
            obs.avoided = True
            return True
        else:
            return False

    # Returns True whenever the dino is jumping over or crosing over
    # the obstacle obs.
    def is_avoiding(self, obs):
        return self.rect.colliderect(obs.reward_rect) and not self.has_crashed

    # Adds the dino's reward to its score if the reward is positive
    def update_score(self):
        if self.reward > 0:
            self.score += self.reward

    # Sets the dino's reward value to be the given value
    def give(self, reward):
        self.reward = reward

    # Updates the dino's speed, position, and sprite
    def update(self, game_speed):
        self.game_speed = game_speed

        if self.is_jumping and not self.has_crashed:
            self.movement[1] = self.movement[1] + self.gravity

        self.rect = self.rect.move(self.movement)
        self.check_bounds()

        self.update_sprites()
        self.counter = (self.counter + 1) % 5

    # Resets the dino to its original state
    def reset(self, game_speed):
        # Resets dino's size, position, and movement
        self.rect.height = self.height
        self.rect.width = self.width
        self.rect.bottom = self.ground_pos
        self.movement = [0, 0]
        self.game_speed = game_speed

        # Resets dino's status
        self.is_jumping = False
        self.is_ducking = False
        self.has_crashed = False

        # Resets the dino's score and reward
        self.score = 0
        self.reward = 0
