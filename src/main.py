import sys
from utils import (
    read_instance,
    initialize_population,
    evaluate_fitness,
    tournament_selection,
    select_parents,
)


def main():

    # Parameters
    POP_SIZE = 100
    GENERATIONS = 500
    ELITE_SIZE = 1
    TOURNAMENT_SIZE = 3

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
    # Main Loop
    for generation in range(GENERATIONS):

        # Evaluate fitness for population
        fitness_results = [evaluate_fitness(individual, E) for individual in population]

        fitnesses = [f[0] for f in fitness_results]
        hard_violations = [f[1] for f in fitness_results]
        soft_violations = [f[2] for f in fitness_results]

        # Best solution found so far
        for i, fit in enumerate(fitnesses):
            if fit < best_fitness:
                best_fitness = fit
                best_solution = population[i]

        # Print progress
        best_idx = fitnesses.index(min(fitnesses))
        print(
            f"Gen {generation + 1}   |   "
            f"Best fitness: {fitnesses[best_idx]}   ||   "
            f"Hard: {hard_violations[best_idx]}     |   "
            f"Soft: {soft_violations[best_idx]}     ||   "
            f"Solution : {population[best_idx]}"
        )

        # Selection
        population = select_parents(
            population,
            fitnesses,
            elite_size=ELITE_SIZE,
            tournament_size=TOURNAMENT_SIZE,
        )

    print("\n=== BEST SOLUTION FOUND ===")
    print("Solution (exam -> slot):", best_solution)
    print("Fitness:", best_fitness)


if __name__ == "__main__":
    main()


# look at calculating violations per student instead of in terms of exams?
