import csv
import random
import numpy as np

def read_jsp(filename):
    jobs = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            job = []
            for task in row:
                machine, processing_time = map(float, task.split(','))  # Change int to float
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

def adam_optimization(problem_index, initial_schedule, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=1000):
    current_schedule = np.array(initial_schedule)  # Convert to NumPy array
    m = np.zeros_like(current_schedule)
    v = np.zeros_like(current_schedule)
    t = 0

    print(f"Solving problem {problem_index}...")
    while t < max_iterations:
        t += 1
        gradients = generate_neighbour(current_schedule)  # Use any method to compute gradients
        gradients = np.array(gradients)  # Convert to NumPy array
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        current_schedule -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    print(f"Best schedule for problem {problem_index}:")
    for job_idx, job in enumerate(current_schedule, start=1):
        print(f"Job {job_idx}: {job}")
    print(f"Completion time: {objective_function(current_schedule)}\n")

# Parameters
num_problems = 100  # Number of JSP problems
problem_filenames = [f"problem_{i}.csv" for i in range(1, num_problems + 1)]

# Solve each problem using Adam Optimization
for i, filename in enumerate(problem_filenames, start=1):
    initial_schedule = read_jsp(filename)
    adam_optimization(i, initial_schedule)
