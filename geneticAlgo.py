import numpy as np
import random
import copy
import csv

class GeneticScheduler:
    def __init__(self, jobs, population_size=50, generations=100, mutation_rate=0.1, crossover_rate=0.8):
        self.jobs = jobs
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_jobs = len(jobs)
        self.num_machines = len(jobs[0])
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            schedule = []
            for job in self.jobs:
                random_schedule = random.sample(job, len(job))
                schedule.append(random_schedule)
            population.append(schedule)
        return population
    
    def calculate_completion_time(self, schedule):
        machine_timings = [0] * self.num_machines
        for tasks in schedule:
            for machine, processing_time in tasks:
                start_time = max(machine_timings[machine - 1], machine_timings[machine - 1] + processing_time)
                machine_timings[machine - 1] = start_time
        return max(machine_timings)
    
    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.num_jobs - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def mutate(self, schedule):
        for idx, job in enumerate(schedule):
            if random.random() < self.mutation_rate:
                mutation_point1, mutation_point2 = random.sample(range(len(job)), 2)
                schedule[idx][mutation_point1], schedule[idx][mutation_point2] = \
                    schedule[idx][mutation_point2], schedule[idx][mutation_point1]
        return schedule
    
    def evolve_population(self, population):
        new_population = []
        ranked_population = sorted(population, key=lambda x: self.calculate_completion_time(x))
        elite_size = int(self.population_size * 0.1) # 상위 10%는 엘리트로 보존
        new_population.extend(ranked_population[:elite_size])
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(ranked_population, 2)
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
        
        return new_population
    
    def find_optimal_schedule(self):
        population = self.initialize_population()
        for _ in range(self.generations):
            population = self.evolve_population(population)
        return min(population, key=lambda x: self.calculate_completion_time(x))

def load_from_csv(filename):
    jobs = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            job = []
            for task_str in row:
                machine, processing_time = map(int, task_str.split(','))
                job.append((machine, processing_time))
            jobs.append(job)
    return jobs

# Load JSP problems
problems = []
for i in range(100):
    filename = f"problem_{i+1}.csv"
    problem = load_from_csv(filename)
    problems.append(problem)

# Solve each problem using Genetic Algorithm
for i, problem in enumerate(problems):
    print(f"Solving problem {i+1}...")
    scheduler = GeneticScheduler(problem)
    best_schedule = scheduler.find_optimal_schedule()
    print(f"Best schedule for problem {i+1}:")
    for idx, tasks in enumerate(best_schedule, start=1):
        print(f"Job {idx}: {tasks}")
    print(f"Completion time: {scheduler.calculate_completion_time(best_schedule)}")
    print()
