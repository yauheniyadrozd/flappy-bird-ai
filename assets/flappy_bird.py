import pygame
import sys
import random
import os

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
FLAP_STRENGTH = -7
PIPE_SPEED = 3
PIPE_GAP = 150
PIPE_FREQUENCY = 1500  # milliseconds

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')
clock = pygame.time.Clock()

# Font
font = pygame.font.Font(os.path.join('assets', '04B_19.ttf'), 30)


class Bird:
    def __init__(self):
        # Load bird images
        self.downflap = pygame.image.load(os.path.join('assets', 'bluebird-downflap.png')).convert_alpha()
        self.midflap = pygame.image.load(os.path.join('assets', 'bluebird-midflap.png')).convert_alpha()
        self.upflap = pygame.image.load(os.path.join('assets', 'bluebird-upflap.png')).convert_alpha()

        # Scale images if needed (assuming original is too large)
        scale_factor = 0.5
        self.downflap = pygame.transform.scale(self.downflap,
                                               (int(self.downflap.get_width() * scale_factor),
                                                int(self.downflap.get_height() * scale_factor)))
        self.midflap = pygame.transform.scale(self.midflap,
                                              (int(self.midflap.get_width() * scale_factor),
                                               int(self.midflap.get_height() * scale_factor)))
        self.upflap = pygame.transform.scale(self.upflap,
                                             (int(self.upflap.get_width() * scale_factor),
                                              int(self.upflap.get_height() * scale_factor)))

        self.images = [self.downflap, self.midflap, self.upflap]
        self.image_index = 0
        self.image = self.images[self.image_index]
        self.rect = self.image.get_rect(center=(100, SCREEN_HEIGHT // 2))

        self.movement = 0
        self.flap_animation_counter = 0

    def update(self):
        # Apply gravity
        self.movement += GRAVITY
        self.rect.y += self.movement

        # Animation
        self.flap_animation_counter += 1
        if self.flap_animation_counter > 5:
            self.flap_animation_counter = 0
            self.image_index = (self.image_index + 1) % 3
            self.image = self.images[self.image_index]

        # Rotate bird
        self.image = pygame.transform.rotate(self.images[self.image_index], -self.movement * 3)

    def flap(self):
        self.movement = FLAP_STRENGTH

    def get_mask(self):
        return pygame.mask.from_surface(self.image)


class Pipe:
    def __init__(self):
        self.pipe_img = pygame.image.load(os.path.join('assets', 'pipe-green.png')).convert_alpha()
        scale_factor = 1.5
        self.pipe_img = pygame.transform.scale(self.pipe_img,
                                               (int(self.pipe_img.get_width() * scale_factor),
                                                int(self.pipe_img.get_height() * scale_factor)))

        self.top_pipe = pygame.transform.flip(self.pipe_img, False, True)
        self.bottom_pipe = self.pipe_img

        self.passed = False
        self.set_position()

    def set_position(self):
        self.y_pos = random.randint(200, 400)
        self.top_rect = self.top_pipe.get_rect(midbottom=(SCREEN_WIDTH + 100, self.y_pos - PIPE_GAP // 2))
        self.bottom_rect = self.bottom_pipe.get_rect(midtop=(SCREEN_WIDTH + 100, self.y_pos + PIPE_GAP // 2))

    def update(self):
        self.top_rect.x -= PIPE_SPEED
        self.bottom_rect.x -= PIPE_SPEED

    def draw(self, screen):
        screen.blit(self.top_pipe, self.top_rect)
        screen.blit(self.bottom_pipe, self.bottom_rect)

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.top_pipe)
        bottom_mask = pygame.mask.from_surface(self.bottom_pipe)

        top_offset = (self.top_rect.x - bird.rect.x, self.top_rect.y - bird.rect.y)
        bottom_offset = (self.bottom_rect.x - bird.rect.x, self.bottom_rect.y - bird.rect.y)

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        return t_point or b_point


class Game:
    def __init__(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.high_score = 0
        self.game_active = False

        # Load background
        self.bg = pygame.image.load(os.path.join('assets', 'background-day.png')).convert()
        self.bg = pygame.transform.scale(self.bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # Load ground
        self.ground_img = pygame.image.load(os.path.join('assets', 'base.png')).convert()
        self.ground_img = pygame.transform.scale(self.ground_img, (SCREEN_WIDTH * 2, self.ground_img.get_height()))
        self.ground_x = 0

        # Load messages
        self.game_over_img = pygame.image.load(os.path.join('assets', 'gameover.png')).convert_alpha()
        self.game_over_rect = self.game_over_img.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))

        self.message_img = pygame.image.load(os.path.join('assets', 'message.png')).convert_alpha()
        self.message_rect = self.message_img.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))

        # Pipe timer
        self.pipe_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.pipe_timer, PIPE_FREQUENCY)

    def draw_ground(self):
        screen.blit(self.ground_img, (self.ground_x, SCREEN_HEIGHT - self.ground_img.get_height()))
        screen.blit(self.ground_img, (self.ground_x + SCREEN_WIDTH, SCREEN_HEIGHT - self.ground_img.get_height()))

        self.ground_x -= PIPE_SPEED
        if self.ground_x <= -SCREEN_WIDTH:
            self.ground_x = 0

    def check_collision(self):
        # Check if bird hits ground or ceiling
        if self.bird.rect.bottom >= SCREEN_HEIGHT - self.ground_img.get_height() or self.bird.rect.top <= 0:
            return True

        # Check pipe collisions
        for pipe in self.pipes:
            if pipe.collide(self.bird):
                return True

        return False

    def update_score(self):
        for pipe in self.pipes:
            if not pipe.passed and pipe.top_rect.right < self.bird.rect.left:
                pipe.passed = True
                self.score += 0.5  # 0.5 because each pair counts as one

        # Update high score
        if self.score > self.high_score:
            self.high_score = self.score

    def draw_score(self, game_active):
        if game_active:
            score_surface = font.render(f'{int(self.score)}', True, WHITE)
            score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, 50))
            screen.blit(score_surface, score_rect)
        else:
            # Game over screen
            score_surface = font.render(f'Score: {int(self.score)}', True, WHITE)
            score_rect = score_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            screen.blit(score_surface, score_rect)

            high_score_surface = font.render(f'High Score: {int(self.high_score)}', True, WHITE)
            high_score_rect = high_score_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100))
            screen.blit(high_score_surface, high_score_rect)

    def reset_game(self):
        self.pipes.clear()
        self.bird.rect.center = (100, SCREEN_HEIGHT // 2)
        self.bird.movement = 0
        self.score = 0
        self.game_active = True

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and self.game_active:
                        self.bird.flap()

                    if event.key == pygame.K_SPACE and not self.game_active:
                        self.reset_game()

                if event.type == self.pipe_timer and self.game_active:
                    self.pipes.append(Pipe())

            # Draw background
            screen.blit(self.bg, (0, 0))

            if self.game_active:
                # Bird
                self.bird.update()
                screen.blit(self.bird.image, self.bird.rect)

                # Pipes
                for pipe in self.pipes:
                    pipe.update()
                    pipe.draw(screen)

                    # Remove off-screen pipes
                    if pipe.top_rect.right < 0:
                        self.pipes.remove(pipe)

                # Check collisions
                if self.check_collision():
                    self.game_active = False

                # Update score
                self.update_score()
            else:
                # Game over screen
                screen.blit(self.message_img, self.message_rect)
                screen.blit(self.game_over_img, self.game_over_rect)

            # Ground
            self.draw_ground()

            # Score
            self.draw_score(self.game_active)

            pygame.display.update()
            clock.tick(60)


if __name__ == "__main__":
    game = Game()
    game.run()