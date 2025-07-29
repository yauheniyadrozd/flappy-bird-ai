import pygame
import sys
import random
import os
import neat
import pickle
import matplotlib.pyplot as plt
import graphviz as gv

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
FLAP_STRENGTH = -7
PIPE_SPEED = 3
PIPE_GAP = 150
PIPE_FREQUENCY = 1500

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird AI')
clock = pygame.time.Clock()

# Font
font = pygame.font.Font(os.path.join('assets', '04B_19.ttf'), 30)


class Bird:
    def __init__(self):
        # Load bird images
        self.downflap = pygame.image.load(os.path.join('assets', 'bluebird-downflap.png')).convert_alpha()
        self.midflap = pygame.image.load(os.path.join('assets', 'bluebird-midflap.png')).convert_alpha()
        self.upflap = pygame.image.load(os.path.join('assets', 'bluebird-upflap.png')).convert_alpha()

        # Scale images
        scale_factor = 1.0
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
        self.alive = True
        self.fitness = 0
        self.time_alive = 0

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

        # Rotate bird based on movement
        self.image = pygame.transform.rotate(self.images[self.image_index], -self.movement * 3)

        # Update fitness
        self.time_alive += 1
        self.fitness += 0.1  # Small fitness reward for staying alive

    def flap(self):
        self.movement = FLAP_STRENGTH

    def get_mask(self):
        return pygame.mask.from_surface(self.image)


class Pipe:
    def __init__(self):
        # Load pipe image
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
        self.birds = []
        self.pipes = []
        self.generation = 0
        self.high_score = 0

        # Load assets
        self.bg = pygame.image.load(os.path.join('assets', 'background-day.png')).convert()
        self.bg = pygame.transform.scale(self.bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.ground_img = pygame.image.load(os.path.join('assets', 'base.png')).convert()
        self.ground_img = pygame.transform.scale(self.ground_img, (SCREEN_WIDTH * 2, self.ground_img.get_height()))
        self.ground_x = 0

        # Pipe timer
        self.pipe_timer = pygame.USEREVENT + 1
        pygame.time.set_timer(self.pipe_timer, PIPE_FREQUENCY)

    def draw_ground(self):
        screen.blit(self.ground_img, (self.ground_x, SCREEN_HEIGHT - self.ground_img.get_height()))
        screen.blit(self.ground_img, (self.ground_x + SCREEN_WIDTH, SCREEN_HEIGHT - self.ground_img.get_height()))
        self.ground_x -= PIPE_SPEED
        if self.ground_x <= -SCREEN_WIDTH:
            self.ground_x = 0

    def check_collision(self, bird):
        # Check if bird hits ground or ceiling
        if bird.rect.bottom >= SCREEN_HEIGHT - self.ground_img.get_height() or bird.rect.top <= 0:
            bird.alive = False

        # Check pipe collisions
        for pipe in self.pipes:
            if pipe.collide(bird):
                bird.alive = False

    def update_score(self, bird):
        for pipe in self.pipes:
            if not pipe.passed and pipe.top_rect.right < bird.rect.left:
                pipe.passed = True
                bird.fitness += 5  # Bigger reward for passing pipes

                # Update high score
                if bird.fitness > self.high_score:
                    self.high_score = bird.fitness

    def draw_score(self):
        score_surface = font.render(f'Gen: {self.generation}', True, WHITE)
        screen.blit(score_surface, (10, 10))

        alive_surface = font.render(f'Alive: {len([b for b in self.birds if b.alive])}', True, WHITE)
        screen.blit(alive_surface, (10, 50))

        high_score_surface = font.render(f'High Score: {int(self.high_score)}', True, WHITE)
        screen.blit(high_score_surface, (10, 90))

    def reset_game(self):
        self.pipes.clear()
        self.generation = 1

    def run(self, genomes, config):
        self.reset_game()

        # Create neural networks and birds
        nets = []
        self.birds = []

        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            self.birds.append(Bird())

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == self.pipe_timer and len(self.birds) > 0:
                    self.pipes.append(Pipe())

            # Find the closest pipe
            closest_pipe = None
            for pipe in self.pipes:
                if pipe.top_rect.right > self.birds[0].rect.left:
                    closest_pipe = pipe
                    break

            # Draw background
            screen.blit(self.bg, (0, 0))

            # Update and draw pipes
            for pipe in self.pipes:
                pipe.update()
                pipe.draw(screen)

                # Remove off-screen pipes
                if pipe.top_rect.right < 0:
                    self.pipes.remove(pipe)

            # Update and draw birds
            for i, bird in enumerate(self.birds):
                if bird.alive:
                    # Get inputs for neural network
                    if closest_pipe:
                        inputs = (
                            bird.rect.y / SCREEN_HEIGHT,
                            (closest_pipe.top_rect.bottom - bird.rect.y) / SCREEN_HEIGHT,
                            (closest_pipe.bottom_rect.top - bird.rect.y) / SCREEN_HEIGHT,
                            (closest_pipe.top_rect.left - bird.rect.right) / SCREEN_WIDTH,
                            bird.movement / 10
                        )

                        # Get output from neural network
                        output = nets[i].activate(inputs)

                        # Decide to flap or not
                        if output[0] > 0.5:
                            bird.flap()

                    bird.update()
                    screen.blit(bird.image, bird.rect)

                    # Check collision and update score
                    self.check_collision(bird)
                    if bird.alive:
                        self.update_score(bird)
                    else:
                        # Small penalty for dying
                        genomes[i][1].fitness = bird.fitness - 1

            # Draw ground
            self.draw_ground()

            # Draw score
            self.draw_score()

            pygame.display.update()
            clock.tick(60)

            # Check if all birds are dead
            if all(not bird.alive for bird in self.birds):
                break


def run_neat(config_path):

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)

    # Add reporters to the population
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Start the evolution
    winner = p.run(Game().run, 10)

    # Save the winner genome
    with open('saved_models/best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    with open('saved_models/stats.pkl', 'wb') as f:
        pickle.dump(stats, f)

    # Generate plots
    plot_stats(stats)

    return winner


def plot_stats(stats):
    plt.figure(figsize=(12, 8))

    # Preparing data
    generations = range(len(stats.most_fit_genomes))
    max_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = stats.get_fitness_mean()
    species_counts = stats.get_species_sizes()

    # Plot 1: Max and Average Fitness
    plt.subplot(2, 2, 1)
    plt.plot(generations, max_fitness, 'b-', label="Max Fitness")
    plt.plot(generations, avg_fitness, 'r-', label="Avg Fitness")
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    # Plot 2: Number of Species
    plt.subplot(2, 2, 2)
    plt.plot(generations, [len(s) for s in species_counts], 'g-')
    plt.title("Number of Species")
    plt.xlabel("Generation")
    plt.ylabel("Species Count")
    # Plot 3: Population Size
    plt.subplot(2, 2, 3)
    plt.plot(generations, [sum(s) for s in species_counts], 'm-')
    plt.title("Population Size")
    plt.xlabel("Generation")
    plt.ylabel("Size")
    # Plot 4: Genome Complexity
    plt.subplot(2, 2, 4)
    avg_nodes = [len(genome.nodes) for genome in stats.most_fit_genomes]
    avg_conns = [len(genome.connections) for genome in stats.most_fit_genomes]
    plt.plot(generations, avg_nodes, 'c-', label="Avg Nodes")
    plt.plot(generations, avg_conns, 'y-', label="Avg Connections")
    plt.title("Genome Complexity")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.legend()
    plt.subplot(2, 2, 1)
    plt.plot(generations, max_fitness, 'b-', label="Max Fitness")
    plt.plot(generations, avg_fitness, 'r-', label="Avg Fitness")
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()


    plt.subplot(2, 2, 2)
    plt.plot(generations, [len(s) for s in species_counts], 'g-')
    plt.title("Number of Species")
    plt.xlabel("Generation")
    plt.ylabel("Species Count")


    plt.subplot(2, 2, 3)
    plt.plot(generations, [sum(s) for s in species_counts], 'm-')
    plt.title("Population Size")
    plt.xlabel("Generation")
    plt.ylabel("Size")


    plt.subplot(2, 2, 4)
    avg_nodes = [len(genome.nodes) for genome in stats.most_fit_genomes]
    avg_conns = [len(genome.connections) for genome in stats.most_fit_genomes]
    plt.plot(generations, avg_nodes, 'c-', label="Avg Nodes")
    plt.plot(generations, avg_conns, 'y-', label="Avg Connections")
    plt.title("Genome Complexity")
    plt.xlabel("Generation")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.savefig('docs/learning_stats.png')
    plt.close()


if __name__ == "__main__":
    # Path to NEAT config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')

    # Create config file if it doesn't exist
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write("""[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000
pop_size              = 100
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 5
num_outputs             = 1

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
""")

    run_neat(config_path)