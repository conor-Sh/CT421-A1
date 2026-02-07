import os
import sys
import random
import statistics
import time


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import (
    read_instance,
    initialize_population,
    evaluate_fitness,
    tournament_selection,
    crossover,
    mutate,
)


def run_ga(instance_file, seed):
    random.seed(seed)

    # -------------------------------
    # PARAMETERS
    # -------------------------------
    POP_SIZE = 400
    GENERATIONS = 300
    ELITE_SIZE = 1
    TOURNAMENT_SIZE = 8
    BASE_MUTATION_RATE = 0.1
    MUTATION_RATE_BOOST = 0.3
    STAGNATION_THRESHOLD = 50
    CROSSOVER_RATE = 0.9

    N, K, M, E = read_instance(instance_file)

    # -------------------------------
    # INITIAL POPULATION
    # -------------------------------
    population = initialize_population(POP_SIZE, N=N, K=K)

    # Fitness is ALWAYS a tuple: (hard, soft)
    best_solution = population[0][:]
    best_fitness = evaluate_fitness(best_solution, E, K)

    generations_since_improvement = 0
    current_mutation_rate = BASE_MUTATION_RATE

    # -------------------------------
    # MAIN GA LOOP
    # -------------------------------
    for generation in range(GENERATIONS):
        fitnesses = [evaluate_fitness(individual, E, K) for individual in population]

        gen_best_idx = fitnesses.index(min(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]

        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_idx][:]
            generations_since_improvement = 0
            current_mutation_rate = BASE_MUTATION_RATE
        else:
            generations_since_improvement += 1
            if generations_since_improvement >= STAGNATION_THRESHOLD:
                current_mutation_rate = BASE_MUTATION_RATE + MUTATION_RATE_BOOST

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

    return best_fitness


def std_dev(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python tests/run_stats.py <data_file> [runs]")
        return

    instance_file = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) == 3 else 30

    hard_values = []
    soft_values = []
    run_times = []

    seeds = [random.randint(0, 2**32 - 1) for _ in range(runs)]
    for i, seed in enumerate(seeds, start=1):
        print(f"Run {i}/{runs} | Seed: {seed}")
        start_time = time.perf_counter()
        best_fitness = run_ga(instance_file, seed)
        elapsed = time.perf_counter() - start_time
        run_times.append(elapsed)
        print(f"  Best Fitness (H,S): {best_fitness[0]}, {best_fitness[1]}")
        print(f"  Runtime: {elapsed:.2f} seconds")
        hard_values.append(best_fitness[0])
        soft_values.append(best_fitness[1])
    print("\nAll runs complete. Calculating summary...\n")

    hard_mean = statistics.mean(hard_values)
    soft_mean = statistics.mean(soft_values)
    hard_best = min(hard_values)
    soft_best = min(soft_values)
    hard_worst = max(hard_values)
    soft_worst = max(soft_values)
    hard_std = std_dev(hard_values)
    soft_std = std_dev(soft_values)
    avg_runtime = statistics.mean(run_times)

    zero_hard_pct = (sum(1 for h in hard_values if h == 0) / runs) * 100

    print("\n=== GA RUN SUMMARY ===")
    print(f"Instance: {instance_file}")
    print(f"Runs: {runs}")
    print(f"Seeds: {', '.join(str(s) for s in seeds)}")

    print("\nHard Violations")
    print(f"Mean: {hard_mean:.3f}")
    print(f"Best (min): {hard_best}")
    print(f"Worst (max): {hard_worst}")
    print(f"Std Dev: {hard_std:.3f}")

    print("\nSoft Violations")
    print(f"Mean: {soft_mean:.3f}")
    print(f"Best (min): {soft_best}")
    print(f"Worst (max): {soft_worst}")
    print(f"Std Dev: {soft_std:.3f}")

    print(f"\n% Runs with 0 hard constraints: {zero_hard_pct:.1f}%")
    print(f"Average runtime per run: {avg_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
