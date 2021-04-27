import os
import random
import pygame


# Class for the ground in the background
class Ground():
    # Class constructor
    def __init__(self, game_params):
        # Gets the ground's images and rects
        self.img_name = game_params['ground_img']

        self.image_1 = self.load_image()
        self.rect_1 = self.image_1.get_rect()

        self.image_2 = self.load_image()
        self.rect_2 = self.image_2.get_rect()

        # Ensures that the ground images are at the correct positions
        self.rect_1.bottom = game_params['ground_pos']
        self.rect_2.bottom = game_params['ground_pos']

        self.rect_2.left = self.rect_1.right

        # Sets the ground's movement
        self.movement = [-game_params['start_speed'], 0]

    # Returns the image for the ground
    def load_image(self):
        path = os.path.join('game_classes/sprites', self.img_name)
        image = pygame.image.load(path).convert()
        colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pygame.RLEACCEL)
        return image

    # Draws the ground on the screen
    def draw(self, screen):
        screen.blit(self.image_1, self.rect_1)
        screen.blit(self.image_2, self.rect_2)

    # Updates the ground's speed, position, and sprites
    def update(self, game_speed):
        # updates the obstacle's speed
        self.movement[0] = -game_speed

        self.rect_1 = self.rect_1.move(self.movement)
        self.rect_2 = self.rect_2.move(self.movement)

        if self.rect_1.right < 0:
            self.rect_1.left = self.rect_2.right

        if self.rect_2.right < 0:
            self.rect_2.left = self.rect_1.right


# Class for the background clouds
class Cloud(pygame.sprite.Sprite):
    # Class constructor
    def __init__(self, game_params, game_speed):
        pygame.sprite.Sprite.__init__(self, self.containers)

        self.width = game_params['cloud_width']
        self.height = game_params['cloud_height']
        self.y_pos = random.randrange(game_params['scr_height']/5,
                                      game_params['scr_height']/2)
        self.img_name = game_params['cloud_img']

        # Gets the cloud's image and rect
        self.image = self.load_sprite()
        self.rect = self.image.get_rect()

        # Ensures that the cloud's image is at the correct position
        self.rect.left = game_params['scr_width']
        self.rect.bottom = self.y_pos

        # Sets the cloud's speed to be a random number
        speed = random.randrange(game_speed - 3, game_speed + 3)
        speed = speed if speed > 0 else 1
        self.movement = [-speed, 0]

    # Draws the cloud on the screen
    def draw(self, screen):
        screen.blit(self.image, self.rect)

    # Returns an image for the clouds
    def load_sprite(self):
        path = os.path.join('game_classes/sprites', self.img_name)
        image = pygame.image.load(path).convert()
        colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pygame.RLEACCEL)
        sprite_dims = (self.width, self.height)
        image = pygame.transform.scale(image, sprite_dims)
        return image

    # Updates the cloud's position
    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()


# Class for the game's scoreboard
class Scoreboard():
    # Class constructor
    def __init__(self, game_params):
        self.score = 0
        self.img_name = game_params['numbers_img']
        self.sprite_num = 12  # The number of characters in sprite sheet
        self.width = 11  # The width of a digit's sprite
        self.height = 13  # The height of a digit's sprite
        self.min_digits = 5  # The minimum number of digits on the scoreboard

        self.background_col = game_params['background_col']

        # Gets the scoreboard's images and rects
        self.image = pygame.Surface((self.min_digits * self.width,
                                     self.height))
        self.rect = self.image.get_rect()

        self.temp_images = self.load_sprites()
        self.temp_rect = self.temp_images[0].get_rect()

        # Ensures that the scoreboard's image is at the correct position
        self.rect.left = game_params['scr_width'] * 0.89
        self.rect.top = game_params['scr_height'] * 0.1

    # Returns a list of images corresponding to the
    # scoreboard's sprites.
    def load_sprites(self):
        path = os.path.join('game_classes/sprites', self.img_name)
        sheet = pygame.image.load(path).convert()

        sheet_rect = sheet.get_rect()

        sprites = []

        size_x = sheet_rect.width/self.sprite_num
        size_y = sheet_rect.height

        for i in range(0, int(self.sprite_num)):
            rect = pygame.Rect((i*size_x, 0, size_x, size_y))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet, (0, 0), rect)

            colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)

            sprite_dims = (self.width, self.height)
            image = pygame.transform.scale(image, sprite_dims)

            sprites.append(image)

        return sprites

    # Draws the scoreboard on the screen
    def draw(self, screen):
        screen.blit(self.image, self.rect)

    # Helper function to get the digits from the sprites
    def extract_digits(self, number):
        digits = []
        while(number > 0):
            remainder = number % 10
            digits.append(int(remainder))
            number = (number-remainder)/10

        length = len(digits)
        # Appends zeros to ensure that number of digits
        # on the screen is at least the minimum
        if length < self.min_digits:
            for i in range(length, int(self.min_digits)):
                digits.append(0)
        # Makes the scoreboard image larger if the number
        # of digits is more than the minimum
        elif len(digits) > self.min_digits:
            self.image = pygame.Surface((length * self.width,
                                         self.height))
            self.rect = self.image.get_rect()
        digits.reverse()
        return digits

    # Updates the scoreboard
    def update(self, score):
        score_digits = self.extract_digits(score)
        self.image.fill(self.background_col)
        for s in score_digits:
            self.image.blit(self.temp_images[s], self.temp_rect)
            self.temp_rect.left += self.temp_rect.width
        self.temp_rect.left = 0
