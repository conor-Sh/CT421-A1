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


def evaluate_fitness(solution, E, hard_weight=1000):
    """
    Evaluates the fitness of a specific timetable solution

    Fitness = hard weight * # hard violations + soft weight * # soft violations

    Hard violation is when student has conflicting exams
    Soft violation is when student has consecutive exams

    Args:
        solution (list): assignments of exams to slots
        E (list of lists): enrollment matrix
        hard_weight (int): penalty for hard constraint violations

    Returns:
        tuple: (fitness, hard_violations, soft_violations)
    """
    hard_violations = 0
    soft_violations = 0

    M = len(E)
    N = len(solution)

    for student in range(M):
        # collect slots of exams that student is enrolled in
        slots = []
        for exam in range(N):
            if E[student][exam] == 1:
                slots.append(solution[exam])

        # Hard constraint
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                if slots[i] == slots[j]:
                    hard_violations += 1

        # Soft constraint 1: consecutive exams
        for i in range(len(slots) - 1):
            if slots[i + 1] == slots[i] + 1:
                soft_violations += 1

    # Lexicographic fitness: (hard_violations, soft_violations)
    fitness = (hard_violations, soft_violations)
    return fitness, hard_violations, soft_violations


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

    # Elitism
    # sort population by ascending fitness
    sorted_indeces = sorted(range(pop_size), key=lambda i: fitnesses[i])
    new_population = [population[i] for i in sorted_indeces[:elite_size]]

    # Tournament Selection
    while len(new_population) < pop_size:
        parent = tournament_selection(population, fitnesses, tournament_size)
        new_population.append(parent)

    return new_population
