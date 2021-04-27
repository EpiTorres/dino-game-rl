import os
import random
import pygame


# Class for all the game's obstacles
class Obstacle(pygame.sprite.Sprite):
    # Class constructor
    def __init__(self, game_params, game_speed):
        self.obs_type = random.randrange(0, 3)
        # Becomes a pterodactyl obstacle
        if (self.obs_type == 0):
            self.create_pterodactyl(game_params)
        # Becomes large cacti obstacle
        elif (self.obs_type == 1):
            self.create_lg_cacti(game_params)
        # Becomes small cacti obstacle
        else:
            self.create_sm_cacti(game_params)

        # Gets the sprites and rect of the obstacle
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.sprites = self.load_sprites()
        self.rect = self.sprites[0].get_rect()
        self.sprite_idx = random.randrange(0, self.sprite_num)
        self.image = self.sprites[self.sprite_idx]
        self.counter = 0

        # Sets the obstacle's position and movement
        self.rect.bottom = self.y_pos
        self.rect.left = game_params['scr_width']
        self.speed = game_speed
        self.movement = [-self.speed, 0]

        # To detect if dino succesfully avoids an obstacle
        self.reward_rect = pygame.Rect((game_params['scr_width'],  # left
                                       0,  # top
                                       self.width,  # width
                                       game_params['scr_height']))  # height
        self.avoided = False

        self.min_gap_coeff = game_params['min_gap_coeff']
        self.max_gap_coeff = game_params['max_gap_coeff']

        # To determine when to create a new obstacle
        self.min_gap = round(self.width * game_speed
                             + self.gap * self.min_gap_coeff)
        self.max_gap = round(self.min_gap * self.max_gap_coeff)

    # Creates a pterodactyl using the parameters in game_params
    def create_pterodactyl(self, game_params):
        idx = random.randrange(0, len(game_params['pter_y_pos']))
        self.y_pos = game_params['pter_y_pos'][idx]
        self.width = game_params['pter_width']
        self.height = game_params['pter_height']
        self.gap = game_params['pter_gap']
        self.sprite_num = 2
        self.sprite_move = True
        self.img_name = game_params['pter_img']

    # Creates large cacti using the parameters in game_params
    def create_lg_cacti(self, game_params):
        length = random.randrange(1, game_params['max_cacti_length']+1)
        self.y_pos = game_params['ground_pos']
        self.width = length * game_params['lg_cacti_width']
        self.height = game_params['lg_cacti_height']
        self.gap = game_params['lg_cacti_gap']
        self.sprite_num = 6 / length
        self.sprite_move = False
        self.img_name = game_params['lg_cacti_img']

    # Creates small cacti using the parameters in game_params
    def create_sm_cacti(self, game_params):
        length = random.randrange(1, game_params['max_cacti_length']+1)
        self.y_pos = game_params['ground_pos']
        self.width = length * game_params['sm_cacti_width']
        self.height = game_params['sm_cacti_height']
        self.gap = game_params['sm_cacti_gap']
        self.sprite_num = 6 / length
        self.sprite_move = False
        self.img_name = game_params['sm_cacti_img']

    # Returns a list of images corresponding to this
    # obstacle's sprites.
    def load_sprites(self):
        # Loads the sprite sheet
        path = os.path.join('game_classes/sprites', self.img_name)
        sheet = pygame.image.load(path).convert()
        sheet_rect = sheet.get_rect()

        # Gets the original dimensions for each sprite
        size_x = sheet_rect.width/self.sprite_num
        size_y = sheet_rect.height

        sprites = []

        # Loops through all sprites in the sprite sheet
        # and appends them to the sprites list
        for i in range(int(self.sprite_num)):
            rect = pygame.Rect((i*size_x, 0, size_x, size_y))

            image = pygame.Surface(rect.size).convert()
            image.blit(sheet, (0, 0), rect)

            colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)

            image = pygame.transform.scale(image, (self.width, self.height))
            sprites.append(image)

        return sprites

    # Update's the min and max gaps between this obstacle and a new
    # obstacle based on this obstacle's speed
    def update_gaps(self):
        self.min_gap = round(self.rect.width * self.speed
                             + self.gap * self.min_gap_coeff)
        self.max_gap = round(self.min_gap * self.max_gap_coeff)

    # Draws the obstacle on the screen
    def draw(self, screen):
        screen.blit(self.image, self.rect)

    # Updates the obstacle's speed, position, and sprite
    def update(self, game_speed):
        # updates the obstacle's speed
        self.speed = game_speed
        self.movement[0] = -self.speed

        # Updates this obstacles sprites
        if self.counter % 10 == 0 and self.sprite_move:
            self.sprite_idx = (self.sprite_idx+1) % self.sprite_num
        self.image = self.sprites[self.sprite_idx]
        self.counter += 1

        # Updates the obstacle's position
        self.rect = self.rect.move(self.movement)
        self.reward_rect = self.reward_rect.move(self.movement)
        self.update_gaps()

        # Removes obstacle from screen if it moves beyond screen
        if self.rect.right < 0:
            self.kill()
