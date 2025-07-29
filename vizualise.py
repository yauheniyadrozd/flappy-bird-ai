import pickle
import matplotlib.pyplot as plt
import neat
import os


def visualize_network(config_path, genome_path):
    # Загрузка конфигурации
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Загрузка генома
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)

    # Визуализация сети
    node_names = {
        -1: 'Y Pos',
        -2: 'Top Pipe',
        -3: 'Bottom Pipe',
        -4: 'Pipe X',
        -5: 'Velocity',
        0: 'Flap?'
    }

    plt.figure(figsize=(15, 10))
    neat.visualize.draw_net(config, genome, True, node_names=node_names)
    plt.title("Best Neural Network Architecture")
    plt.savefig('docs/best_network.png')
    plt.close()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config.txt')
    genome_path = os.path.join(local_dir, 'saved_models/best_genome.pkl')

    visualize_network(config_path, genome_path)