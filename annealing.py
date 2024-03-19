import csv
import random
import math

def read_jsp(filename):
    jobs = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            job = []
            for task in row:
                machine, processing_time = map(int, task.split(','))
                job.append((machine, processing_time))
            jobs.append(job)
    return jobs

def objective_function(schedule):
    makespan = max([sum(task[1] for task in job) for job in schedule])
    return makespan

def generate_neighbour(current_schedule):
    new_schedule = current_schedule[:]
    # Randomly swap two tasks within each job
    for job in new_schedule:
        idx1, idx2 = random.sample(range(len(job)), 2)
        job[idx1], job[idx2] = job[idx2], job[idx1]
    return new_schedule

def simulated_annealing(problem_index, initial_schedule, initial_temperature=100, cooling_rate=0.95, min_temperature=0.01, max_iterations=1000):
    current_schedule = initial_schedule
    current_temperature = initial_temperature
    best_schedule = current_schedule
    best_score = objective_function(current_schedule)
    iteration = 0

    print(f"Solving problem {problem_index}...")
    while current_temperature > min_temperature and iteration < max_iterations:
        neighbour_schedule = generate_neighbour(current_schedule)
        neighbour_score = objective_function(neighbour_schedule)
        delta_score = neighbour_score - best_score

        if delta_score < 0 or random.random() < math.exp(-delta_score / current_temperature):
            current_schedule = neighbour_schedule
            best_score = neighbour_score
            best_schedule = neighbour_schedule

        current_temperature *= cooling_rate
        iteration += 1

    print(f"Best schedule for problem {problem_index}:")
    for job_idx, job in enumerate(best_schedule, start=1):
        print(f"Job {job_idx}: {job}")
    print(f"Completion time: {best_score}\n")

# Parameters
num_problems = 100  # Number of JSP problems
problem_filenames = [f"problem_{i}.csv" for i in range(1, num_problems + 1)]

# Solve each problem using simulated annealing
for i, filename in enumerate(problem_filenames, start=1):
    initial_schedule = read_jsp(filename)
    simulated_annealing(i, initial_schedule)
