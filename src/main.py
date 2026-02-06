import sys
import random
import matplotlib.pyplot as plt
import time
from utils import (
    read_instance,
    initialize_population,
    evaluate_fitness,
    tournament_selection,
    crossover,
    mutate,
)


def main():
    start_time = time.time()
    # -------------------------------
    # PARAMETERS
    # -------------------------------
    POP_SIZE = 400
    GENERATIONS = 300
    ELITE_SIZE = 1
    TOURNAMENT_SIZE = 8
    BASE_MUTATION_RATE = 0.1
    MUTATION_RATE_BOOST = 0.3
    STAGNATION_THRESHOLD = 20
    CROSSOVER_RATE = 0.9

    # -------------------------------
    # INPUT
    # -------------------------------
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python src/main.py <data_file> [seed]")
        return

    instance_file = sys.argv[1]

    # Set seed for reproducibility if provided
    if len(sys.argv) == 3:
        seed = int(sys.argv[2])
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    N, K, M, E = read_instance(instance_file)
    print(f"Exams: {N}, Slots: {K}, Students: {M}")

    # -------------------------------
    # INITIAL POPULATION
    # -------------------------------
    population = initialize_population(POP_SIZE, N=N, K=K)

    # Fitness is ALWAYS a tuple: (hard, soft)
    best_solution = population[0][:]
    best_fitness = evaluate_fitness(best_solution, E, K)

    best_fitness_history = []
    generations_since_improvement = 0
    current_mutation_rate = BASE_MUTATION_RATE

    # -------------------------------
    # MAIN GA LOOP
    # -------------------------------
    for generation in range(GENERATIONS):

        # Evaluate fitness for population
        fitnesses = [evaluate_fitness(individual, E, K) for individual in population]

        # Best in this generation (tuple comparison)
        gen_best_idx = fitnesses.index(min(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]

        # Track best overall + stagnation
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_idx][:]
            generations_since_improvement = 0
            current_mutation_rate = BASE_MUTATION_RATE
        else:
            generations_since_improvement += 1
            if generations_since_improvement >= STAGNATION_THRESHOLD:
                current_mutation_rate = BASE_MUTATION_RATE + MUTATION_RATE_BOOST

        best_fitness_history.append(best_fitness)

        # -------------------------------
        # RANDOM IMMIGRANTS
        # -------------------------------
        immigrant_rate = (
            0.5 if generations_since_improvement > STAGNATION_THRESHOLD else 0.1
        )
        num_immigrants = max(1, round(POP_SIZE * immigrant_rate))

        sorted_indices = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])
        worst_indices = sorted_indices[-num_immigrants:]
        for idx in worst_indices:
            population[idx] = [random.randint(0, K - 1) for _ in range(N)]

        # -------------------------------
        # PROGRESS PRINT
        # -------------------------------
        print(
            f"Gen {generation + 1:3d} | "
            f"Gen Best (H,S): {gen_best_fitness} | "
            f"Overall Best (H,S): {best_fitness} || "
            f"Mutation Rate: {current_mutation_rate:.3f} | "
            f"Stagnant: {generations_since_improvement}/{STAGNATION_THRESHOLD}"
        )

        # -------------------------------
        # ELITISM
        # -------------------------------
        new_population = [population[i][:] for i in sorted_indices[:ELITE_SIZE]]

        # -------------------------------
        # REPRODUCTION
        # -------------------------------
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            if random.random() < CROSSOVER_RATE:
                method = "single_point" if generation % 2 == 0 else "uniform"
                child1, child2 = crossover(parent1, parent2, method=method)
            else:
                child1, child2 = parent1[:], parent2[:]

            child1 = mutate(child1, K, current_mutation_rate)
            child2 = mutate(child2, K, current_mutation_rate)

            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = new_population[:POP_SIZE]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds")

    # -------------------------------
    # BEST OUTPUT
    # -------------------------------
    print("\n=== BEST SOLUTION FOUND ===")
    print("Solution (exam -> slot):", best_solution)
    print("Fitness (hard, soft):", best_fitness)

    # -------------------------------
    # PLOT FITNESS HISTORY
    # -------------------------------
    hard_history = [fit[0] for fit in best_fitness_history]
    soft_history = [fit[1] for fit in best_fitness_history]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hard_history, linewidth=2, label="Hard Violations")
    plt.xlabel("Generation")
    plt.ylabel("Hard Violations")
    plt.title("Hard Constraint Violations Over Generations")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(soft_history, linewidth=2, label="Soft Violations")
    plt.xlabel("Generation")
    plt.ylabel("Soft Violations")
    plt.title("Soft Constraint Violations Over Generations")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("fitness_history.png")
    print("\nPlot saved as 'fitness_history.png'")
    plt.show()


if __name__ == "__main__":
    main()


# need to balance tournament size with population size, keeping elite at 1
# could break down soft contraints by type
