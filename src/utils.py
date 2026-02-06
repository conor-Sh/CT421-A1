import os
import random


def read_instance(filepath):
    """Read an instance file and return N, K, M, E.

    Args:
        filepath (str): path to the instance file

    Returns:
        tuple: (N, K, M, E) where E is a list of M rows, each with N integers
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as file:
        # strip trailing newlines but keep empty-line handling safe
        lines = [ln.rstrip() for ln in file if ln.strip() != ""]

    if not lines:
        raise ValueError(f"Input file is empty: {filepath}")

    try:
        N, K, M = map(int, lines[0].split())
    except Exception as e:
        raise ValueError(f"Could not parse header line in {filepath}: {e}")

    if len(lines) < 1 + M:
        raise ValueError(
            f"Expected {M} enrollment lines, but file has only {len(lines)-1} in {filepath}"
        )

    E = []
    for i in range(1, 1 + M):
        parts = lines[i].split()
        try:
            row = list(map(int, parts))
        except Exception as e:
            raise ValueError(f"Could not parse enrollment row {i} in {filepath}: {e}")

        if len(row) != N:
            raise ValueError(
                f"Expected {N} exams, got {len(row)} exams on line {i+1} in {filepath}"
            )
        E.append(row)

    return N, K, M, E


def initialize_population(pop_size, N, K):
    """Create an initial population of solutions

    Args:
        pop_size (int): number of solutions
        N (int): number of exams
        K (int): number of time slots

    Returns:
        list of lists: population, a list of solutions, each is a list of length N
    """
    population = []

    # Assign random time slot for each exam
    for i in range(pop_size):
        solution = [random.choice(range(K)) for j in range(N)]
        population.append(solution)

    return population


def evaluate_fitness(solution, E, K):
    """
    Fitness is a tuple (hard_violations, soft_violations).

    Hard violations:
        Number of conflicting exam pairs for each student.

    Soft violations:
        1. Consecutive exams for a student (penalty +2 per consecutive pair)
        2. Exams scheduled in the last time slot (penalty +1 each)
    """
    hard_violations = 0
    soft_violations = 0

    M = len(E)
    N = len(solution)

    last_slot = K - 1

    for student in range(M):
        # collect slots of exams the student is enrolled in
        slots = [solution[exam] for exam in range(N) if E[student][exam] == 1]

        # Hard constraint: count conflicting pairs
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                if slots[i] == slots[j]:
                    hard_violations += 1

        # Soft constraint 1: consecutive exams (must sort!)
        slots.sort()
        for i in range(len(slots) - 1):
            if slots[i + 1] == slots[i] + 1:
                soft_violations += 2

        # Soft constraint 2: exams in the last slot
        for slot in slots:
            if slot == last_slot:
                soft_violations += 1

    return (hard_violations, soft_violations)


def tournament_selection(population, fitnesses, tournament_size):
    """
    Select individual from population using tournament selection

    Args:
        population (list): list of solutions
        fitnesses (list): list of solutions' fitness
        tournament_size (int): number of individuals in tournament

    Returns:
        selected solution
    """
    selected_indices = random.sample(range(len(population)), tournament_size)

    best_index = selected_indices[0]
    for index in selected_indices[1:]:
        if fitnesses[index] < fitnesses[best_index]:
            best_index = index

    return population[best_index]


def select_parents(population, fitnesses, elite_size=1, tournament_size=3):
    """
    Select new population using elitism and tournament selection

    Args:
        population (list): list of solutions
        fitnesses (list): list of solutions' fitness
        tournament_size (int): number of individuals in tournament
        elite_size (int): number of elite individuals to preserve

    Returns:
        selected population (list), same size as original
    """

    pop_size = len(population)

    # sort population by ascending fitness
    sorted_indeces = sorted(range(pop_size), key=lambda i: fitnesses[i])
    new_population = [population[i] for i in sorted_indeces[:elite_size]]

    # Tournament Selection
    while len(new_population) < pop_size:
        parent = tournament_selection(population, fitnesses, tournament_size)
        new_population.append(parent)

    return new_population


def crossover(parent1, parent2, method="single_point"):
    """
    Create child from two parent solutions via crossover

    Args:
        parent1 (list): first parent solution (exam -> slot assignments)
        parent2 (list): second parent solution
        method (str): crossover method - "single_point" or "uniform"

    Returns:
        tuple: (child1, child2)
    """
    N = len(parent1)

    if method == "single_point":
        # Single-point crossover: split at random point and swap segments
        crossover_point = random.randint(1, N - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

    elif method == "uniform":
        # Uniform crossover: randomly inherit each gene from either parent
        child1 = []
        child2 = []
        for i in range(N):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])

    else:
        raise ValueError(f"Unknown crossover method: {method}")

    return child1, child2


def mutate(solution, K, mutation_rate):
    """
    Mutate a solution by randomly changing exam slot assignments

    Args:
        solution (list): exam -> slot assignments
        K (int): number of time slots
        mutation_rate (float): probability of mutating each exam

    Returns:
        mutated solution (list)
    """
    mutated = solution[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = random.randint(0, K - 1)
    return mutated


def population_diversity(population):
    # Compute average normalized Hamming distance between all pairs
    pop_size = len(population)
    if pop_size < 2:
        return 0.0
    total_dist = 0
    count = 0
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = sum(a != b for a, b in zip(population[i], population[j]))
            total_dist += dist / len(population[i])
            count += 1
    return total_dist / count if count > 0 else 0.0
