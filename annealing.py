import csv
import random
import math

def read_problem_from_csv(filename):
    jobs = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            job = [(int(machine), int(time)) for machine, time in (task.split(',') for task in row)]
            jobs.append(job)
    return jobs

def calculate_makespan(jobs):
    max_time = 0
    for job in jobs:
        job_time = sum([time for _, time in job])
        if job_time > max_time:
            max_time = job_time
    return max_time

def simulated_annealing_solver(filename):
    jobs = read_problem_from_csv(filename)
    current_solution = jobs
    current_makespan = calculate_makespan(current_solution)
    best_solution = current_solution
    best_makespan = current_makespan
    
    temp = 10
    cooling_rate = 0.3
    
    while temp > 1:
        new_solution = [list(job) for job in current_solution]
        
        # 작업 순서 변경
        for _ in range(int(len(new_solution) * 0.5)):  # 작업 순서를 더 자주 변경
            job1, job2 = random.sample(range(len(new_solution)), 2)
            new_solution[job1], new_solution[job2] = new_solution[job2], new_solution[job1]
        
        # 작업 내의 기계와 시간 할당 순서 변경
        for job in new_solution:
            if random.random() < 0.5:  # 50% 확률로 작업 내의 순서 변경
                random.shuffle(job)
        
        new_makespan = calculate_makespan(new_solution)
        
        if new_makespan < current_makespan or random.random() < math.exp((current_makespan - new_makespan) / temp):
            current_solution = new_solution
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_solution = current_solution
                best_makespan = current_makespan
        
        temp *= 1 - cooling_rate
    
    return best_makespan, best_solution

def print_jobs_allocation(jobs):
    max_machine_number = max(machine for job in jobs for machine, _ in job)
    machine_allocations = {machine: [] for machine in range(1, max_machine_number + 1)}
    
    for job_id, job in enumerate(jobs, start=1):
        for machine, _ in job:
            if machine in machine_allocations:
                machine_allocations[machine].append(job_id)

for i in range(1, 101):
    filename = f"problem_{i}.csv"
    makespan, jobs = simulated_annealing_solver(filename)
    print(f"문제 {i}: 총 처리 시간 = {makespan}")
    print_jobs_allocation(jobs)
