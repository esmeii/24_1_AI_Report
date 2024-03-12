import numpy as np
import random

# Define job shop scheduling problem
# Each job is represented by a list of tuples (task, processing_time)
# Each machine is represented by an integer

jobs = [
    [(1, 3), (2, 2), (3, 2)],
    [(1, 2), (2, 1), (3, 4)],
    [(1, 4), (2, 3)]
]

num_jobs = len(jobs)
num_machines = max(task[0] for job in jobs for task in job)

# Genetic Algorithm parameters
population_size = 50
mutation_rate = 0.1
generations = 100

# Function to calculate makespan
def calculate_makespan(schedule):
    machine_times = [0] * num_machines
    for job in schedule:
        for task in job:
            machine = task[0] - 1
            processing_time = task[1]
            machine_times[machine] += processing_time
    return max(machine_times)

# Function to initialize population
def initialize_population():
    population = []
    for _ in range(population_size):
        schedule = []
        for job in jobs:
            schedule.append(sorted(job, key=lambda x: random.random()))
        population.append(schedule)
    return population

# Function to select parents for crossover
def select_parents(population):
    return random.choices(population, k=2)

# Function to perform crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_jobs - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Function to perform mutation
def mutate(schedule):
    for job in schedule:
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(job) - 1)
            job[mutation_point] = (job[mutation_point][0], random.randint(1, 5))  # Change processing time
    return schedule

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population()
    for _ in range(generations):
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
        best_schedule = min(population, key=calculate_makespan)
        best_makespan = calculate_makespan(best_schedule)
        print(f"Generation {_+1}, Best Makespan: {best_makespan}")

    return best_schedule, best_makespan

best_schedule, best_makespan = genetic_algorithm()
print("Best Schedule:", best_schedule)
print("Best Makespan:", best_makespan)

