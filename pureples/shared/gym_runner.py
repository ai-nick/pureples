import neat 
import numpy as np
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork


# Initialize population attaching statistics reporter.
def ini_pop(state, stats, config, output):
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


# Generic OpenAI Gym runner for ES-HyperNEAT.
def run_es(gens, env, max_steps, config, params, substrate, max_trials=100, output=True):
    trials = 1

    def eval_fitness(genomes, config):

        for idx, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(substrate, cppn, params)
            net = network.create_phenotype_network("es")

            fitnesses = []

            for i in range(trials):
                ob = env.reset()
                net.reset()

                total_reward = 0

                for j in range(max_steps):
                    for k in range(network.activations):
                        o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)
            
            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)  
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)

def run_nd_es(gens, env, max_steps, config, params, substrate, max_trials=100, output=True):
    trials = 1

    def eval_fitness(genomes, config):

        for idx, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(substrate, cppn, params)
            net = network.create_phenotype_network_nd("nd_es")

            fitnesses = []

            for i in range(trials):
                ob = env.reset()
                net.reset()

                total_reward = 0

                for j in range(max_steps):
                    for k in range(network.activations):
                        o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)
            
            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


# Generic OpenAI Gym runner for HyperNEAT.
def run_hyper(gens, env, max_steps, config, substrate, activations, max_trials=100, activation="sigmoid", output=True):
    trials = 1

    def eval_fitness(genomes, config):

        for idx, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            net = create_phenotype_network(cppn, substrate, activation)

            fitnesses = []

            for i in range(trials):
                ob = env.reset()
                net.reset()

                total_reward = 0

                for j in range(max_steps):
                    for k in range(activations):
                        o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)
            
            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


# Generic OpenAI Gym runner for NEAT.
def run_neat(gens, env, max_steps, config, max_trials=100, output=True):
    trials = 1

    def eval_fitness(genomes, config):

        for idx, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)

            fitnesses = []

            for i in range(trials):
                ob = env.reset()

                total_reward = 0

                for j in range(max_steps):
                    o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, info = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)
            
            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)

