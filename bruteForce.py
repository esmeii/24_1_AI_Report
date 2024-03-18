import csv

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

def calculate_completion_time(schedule, jobs):
    num_machines = len(schedule[0])
    machine_timings = [0] * num_machines
    for job_idx, tasks in enumerate(schedule):
        for task_idx, (machine, processing_time) in enumerate(tasks):
            start_time = max(machine_timings[machine - 1], job_idx)
            machine_timings[machine - 1] = start_time + processing_time
    return max(machine_timings)

def brute_force_schedule(jobs):
    num_jobs = len(jobs)
    num_machines = len(jobs[0])
    min_completion_time = float('inf')
    best_schedule = None

    def generate_schedules(schedule, job_index):
        nonlocal min_completion_time, best_schedule
        if job_index == num_jobs:
            completion_time = calculate_completion_time(schedule, jobs)
            if completion_time < min_completion_time:
                min_completion_time = completion_time
                best_schedule = schedule
        else:
            for perm in itertools.permutations(jobs[job_index]):
                new_schedule = schedule + [perm]
                generate_schedules(new_schedule, job_index + 1)

    generate_schedules([], 0)
    return best_schedule

import itertools

# Load JSP problems
problems = []
for i in range(100):
    filename = f"problem_{i+1}.csv"
    problem = load_from_csv(filename)
    problems.append(problem)

# Solve each problem
for i, problem in enumerate(problems):
    print(f"Solving problem {i+1}...")
    best_schedule = brute_force_schedule(problem)
    print(f"Best schedule for problem {i+1}:")
    for idx, tasks in enumerate(best_schedule, start=1):
        print(f"Job {idx}: {tasks}")
    print(f"Completion time: {calculate_completion_time(best_schedule, problem)}")
    print()
