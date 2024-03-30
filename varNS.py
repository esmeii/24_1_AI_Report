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

def shaking(jobs, k):
    new_jobs = [list(job) for job in jobs]
    if k <= len(jobs) / 2:
        # 단순 교환: k 값에 따라 교환하는 작업의 수를 증가시킵니다.
        for _ in range(k):
            job1, job2 = random.sample(range(len(new_jobs)), 2)
            new_jobs[job1], new_jobs[job2] = new_jobs[job2], new_jobs[job1]
    else:
        # 연속적 교환: k 값에 따라 연속적인 작업을 교환합니다.
        start_point = random.randint(0, len(new_jobs) - k)
        end_point = start_point + k
        jobs_sublist = new_jobs[start_point:end_point]
        random.shuffle(jobs_sublist)
        new_jobs[start_point:end_point] = jobs_sublist
    return new_jobs

def local_search(jobs):
    current_solution = jobs
    current_makespan = calculate_makespan(current_solution)
    improved = True
    
    while improved:
        improved = False
        for i in range(len(jobs)):
            for j in range(i + 1, len(jobs)):
                new_solution = [list(job) for job in current_solution]
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_makespan = calculate_makespan(new_solution)
                if new_makespan < current_makespan:
                    current_solution = new_solution
                    current_makespan = new_makespan
                    improved = True
                    break
            if improved:
                break
                
    return current_solution

def variable_neighborhood_search(filename):
    jobs = read_problem_from_csv(filename)
    best_solution = jobs
    best_makespan = calculate_makespan(best_solution)
    
    k_max = 10
    k = 1
    
    while k <= k_max:
        new_solution = shaking(best_solution, k)
        new_solution = local_search(new_solution)
        new_makespan = calculate_makespan(new_solution)
        
        if new_makespan < best_makespan:
            best_solution = new_solution
            best_makespan = new_makespan
            k = 1
        else:
            k += 1
    
    return best_makespan, best_solution

def print_jobs_allocation(jobs):
    max_machine_number = max(machine for job in jobs for machine, _ in job)
    machine_allocations = {machine: [] for machine in range(1, max_machine_number + 1)}
    
    for job_id, job in enumerate(jobs, start=1):
        for machine, _ in job:
            machine_allocations[machine].append(job_id)

for i in range(1, 101):
    filename = f"problem_{i}.csv"
    makespan, jobs = variable_neighborhood_search(filename)
    print(f"문제 {i}: 총 처리 시간 = {makespan}")
    print_jobs_allocation(jobs)
