import os
import sys
import random
import statistics


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


def run_ga(instance_file, seed, base_mutation_rate, crossover_rate, tournament_size):
    random.seed(seed)

    # -------------------------------
    # PARAMETERS
    # -------------------------------
    POP_SIZE = 400
    GENERATIONS = 300
    ELITE_SIZE = 1
    MUTATION_RATE_BOOST = 0.3
    STAGNATION_THRESHOLD = 50

    N, K, M, E = read_instance(instance_file)

    # -------------------------------
    # INITIAL POPULATION
    # -------------------------------
    population = initialize_population(POP_SIZE, N=N, K=K)

    # Fitness is ALWAYS a tuple: (hard, soft)
    best_solution = population[0][:]
    best_fitness = evaluate_fitness(best_solution, E, K)

    generations_since_improvement = 0
    current_mutation_rate = base_mutation_rate

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
            current_mutation_rate = base_mutation_rate
        else:
            generations_since_improvement += 1
            if generations_since_improvement >= STAGNATION_THRESHOLD:
                current_mutation_rate = base_mutation_rate + MUTATION_RATE_BOOST

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
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            if random.random() < crossover_rate:
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


def mean(values):
    return statistics.mean(values) if values else 0.0


def std_dev(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def chunk_seeds(seeds, folds):
    folds = max(2, min(folds, len(seeds)))
    chunked = [[] for _ in range(folds)]
    for i, seed in enumerate(seeds):
        chunked[i % folds].append(seed)
    return chunked


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python tests/param_cv.py <data_file> [runs] [folds]")
        return

    instance_file = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) >= 3 else 30
    folds = int(sys.argv[3]) if len(sys.argv) == 4 else 5

    seeds = [random.randint(0, 2**32 - 1) for _ in range(runs)]
    seed_folds = chunk_seeds(seeds, folds)

    # Search grid for parameters
    base_mutation_rates = [0.05, 0.1, 0.2]
    crossover_rates = [0.7, 0.9]
    tournament_sizes = [4, 8, 12]

    results = []

    print("\n=== PARAMETER CROSS-VALIDATION ===")
    print(f"Instance: {instance_file}")
    print(f"Runs: {runs}")
    print(f"Folds: {len(seed_folds)}")
    print(
        f"Grid: base_mutation_rate={base_mutation_rates}, "
        f"crossover_rate={crossover_rates}, tournament_size={tournament_sizes}"
    )

    for base_mutation_rate in base_mutation_rates:
        for crossover_rate in crossover_rates:
            for tournament_size in tournament_sizes:
                fold_hard_means = []
                fold_soft_means = []
                zero_hard_counts = 0
                total_runs = 0

                print(
                    "\nConfig: "
                    f"base_mutation_rate={base_mutation_rate}, "
                    f"crossover_rate={crossover_rate}, "
                    f"tournament_size={tournament_size}"
                )

                for fold_idx, fold_seeds in enumerate(seed_folds, start=1):
                    hard_values = []
                    soft_values = []

                    print(
                        f"  Fold {fold_idx}/{len(seed_folds)} | Seeds: {len(fold_seeds)}"
                    )
                    for seed in fold_seeds:
                        best_fitness = run_ga(
                            instance_file,
                            seed,
                            base_mutation_rate,
                            crossover_rate,
                            tournament_size,
                        )
                        hard_values.append(best_fitness[0])
                        soft_values.append(best_fitness[1])
                        if best_fitness[0] == 0:
                            zero_hard_counts += 1
                        total_runs += 1

                    fold_hard_means.append(mean(hard_values))
                    fold_soft_means.append(mean(soft_values))

                result = {
                    "base_mutation_rate": base_mutation_rate,
                    "crossover_rate": crossover_rate,
                    "tournament_size": tournament_size,
                    "hard_mean": mean(fold_hard_means),
                    "hard_std": std_dev(fold_hard_means),
                    "soft_mean": mean(fold_soft_means),
                    "soft_std": std_dev(fold_soft_means),
                    "zero_hard_pct": (zero_hard_counts / total_runs) * 100,
                }
                results.append(result)

                print(
                    "  Summary | "
                    f"Hard mean: {result['hard_mean']:.3f} (std {result['hard_std']:.3f}) | "
                    f"Soft mean: {result['soft_mean']:.3f} (std {result['soft_std']:.3f}) | "
                    f"% zero hard: {result['zero_hard_pct']:.1f}%"
                )

    results.sort(key=lambda r: (r["hard_mean"], r["soft_mean"]))

    print("\n=== TOP CONFIGS ===")
    for idx, r in enumerate(results[:5], start=1):
        print(
            f"{idx}. base_mutation_rate={r['base_mutation_rate']}, "
            f"crossover_rate={r['crossover_rate']}, "
            f"tournament_size={r['tournament_size']} | "
            f"Hard mean: {r['hard_mean']:.3f} (std {r['hard_std']:.3f}) | "
            f"Soft mean: {r['soft_mean']:.3f} (std {r['soft_std']:.3f}) | "
            f"% zero hard: {r['zero_hard_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
