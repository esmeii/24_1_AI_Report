import csv
import random

def generate_jsp(num_jobs, num_machines, min_processing_time, max_processing_time):
    jobs = []
    for _ in range(num_jobs):
        job = []
        for machine in range(1, num_machines + 1):
            processing_times = [random.randint(min_processing_time, max_processing_time) for _ in range(num_machines)]
            for idx, processing_time in enumerate(processing_times, start=1):
                job.append((idx, processing_time))
        jobs.append(job)
    return jobs

def save_to_csv(jobs, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for job in jobs:
            writer.writerow([f"{task[0]},{task[1]}" for task in job])

# Parameters
num_problems = 100  # Number of JSP problems to generate
num_jobs = 12      # Numaber of jobs in each problem
num_machines = 5  # Number of machines in each problem
min_processing_time = 1  # Minimum processing time for a task
max_processing_time = 5 # Maximum processing time for a task

# Generate JSP problems
for i in range(0,100):
    jobs = generate_jsp(num_jobs, num_machines, min_processing_time, max_processing_time)
    filename = f"problem_{i+1}.csv"
    save_to_csv(jobs, filename)
    print(f"Problem {i+1} saved as {filename}")
