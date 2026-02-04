import sys
import random
import matplotlib.pyplot as plt
from utils import (
    read_instance,
    initialize_population,
    evaluate_fitness,
    tournament_selection,
    crossover,
    mutate,
)


def main():

    # Parameters
    POP_SIZE = 300
    GENERATIONS = 500
    ELITE_SIZE = 1
    TOURNAMENT_SIZE = 2
    BASE_MUTATION_RATE = 0.03
    MUTATION_RATE_BOOST = 0.2
    STAGNATION_THRESHOLD = 50
    CROSSOVER_RATE = 0.95

    if len(sys.argv) != 2:
        print("Usage: python src/main.py <data_file>")
        return

    instance_file = sys.argv[1]
    N, K, M, E = read_instance(instance_file)

    print(f"Exams: {N}, Slots: {K}, Students: {M}\nEnrollment Matrix:\n{E}")

    # Initial Population
    population = initialize_population(POP_SIZE, N=N, K=K)
    best_solution = None
    best_fitness = (float("inf"), float("inf"))
    best_fitness_history = []
    generations_since_improvement = 0
    current_mutation_rate = BASE_MUTATION_RATE

    # Main Loop
    for generation in range(GENERATIONS):

        # Evaluate fitness for population
        fitness_results = [evaluate_fitness(individual, E) for individual in population]

        fitnesses = [f[0] for f in fitness_results]
        hard_violations = [f[1] for f in fitness_results]
        soft_violations = [f[2] for f in fitness_results]

        # Find best in this generation
        gen_best_idx = fitnesses.index(min(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]

        # Track best overall and detect stagnation
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_idx][:]
            generations_since_improvement = 0
            current_mutation_rate = (
                BASE_MUTATION_RATE  # Reset mutation rate on improvement
            )
        else:
            generations_since_improvement += 1
            # Increase mutation rate if stagnating
            if generations_since_improvement >= STAGNATION_THRESHOLD:
                current_mutation_rate = BASE_MUTATION_RATE + MUTATION_RATE_BOOST

        best_fitness_history.append(best_fitness)

        # Random immigrants: replace worst solutions with random ones
        # 10% normally, 50% when severely stagnating
        if generations_since_improvement > STAGNATION_THRESHOLD * 2:
            immigrant_rate = 0.5
        else:
            immigrant_rate = 0.1

        num_immigrants = max(1, round(POP_SIZE * immigrant_rate))
        sorted_indices = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])
        worst_indices = sorted_indices[-num_immigrants:]
        for idx in worst_indices:
            population[idx] = [random.randint(0, K - 1) for _ in range(N)]

        # Print progress - show best overall and best in generation
        print(
            f"Gen {generation + 1}   |   "
            f"Gen Best: {gen_best_fitness}   |   "
            f"Overall Best: {best_fitness}   ||   "
            f"Hard: {hard_violations[gen_best_idx]}   |   "
            f"Soft: {soft_violations[gen_best_idx]}   |   "
            f"Mutation Rate: {current_mutation_rate:.3f}   |   "
            f"Stagnant: {generations_since_improvement}/{STAGNATION_THRESHOLD}"
        )

        # Keep elite (best overall) and create offspring via crossover
        sorted_indices = sorted(range(POP_SIZE), key=lambda i: fitnesses[i])
        new_population = [population[i][:] for i in sorted_indices[:ELITE_SIZE]]

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, fitnesses, TOURNAMENT_SIZE)

            # Alternate between crossover methods
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

    print("\n=== BEST SOLUTION FOUND ===")
    print("Solution (exam -> slot):", best_solution)
    print("Fitness:", best_fitness)

    # Plot fitness history
    hard_violations_history = [fit[0] for fit in best_fitness_history]
    soft_violations_history = [fit[1] for fit in best_fitness_history]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hard_violations_history, label="Hard Violations", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Hard Violations")
    plt.title("Hard Constraint Violations Over Generations")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        soft_violations_history, label="Soft Violations", linewidth=2, color="orange"
    )
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
