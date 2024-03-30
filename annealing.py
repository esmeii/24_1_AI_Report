import csv
import random
import math

# CSV 파일로부터 문제를 읽는 함수
def read_problem_from_csv(filename):
    jobs = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            job = [(int(machine), int(time)) for machine, time in (task.split(',') for task in row)]
            jobs.append(job)
    return jobs

# 적합도 함수: 총 makespan 계산
def calculate_makespan(jobs):
    max_time = 0
    for job in jobs:
        job_time = sum([time for _, time in job])
        if job_time > max_time:
            max_time = job_time
    return max_time

# Simulated Annealing 알고리즘 구현
def simulated_annealing_solver(filename):
    jobs = read_problem_from_csv(filename)
    current_solution = jobs
    current_makespan = calculate_makespan(current_solution)
    best_solution = current_solution
    best_makespan = current_makespan
    
    # 초기 온도 및 냉각률 설정
    temp = 10000
    cooling_rate = 0.003
    
    while temp > 1:
        # 새로운 해 생성 (여기서는 간단히 두 작업을 무작위로 교환)
        new_solution = [list(job) for job in current_solution]
        job1, job2 = random.sample(range(len(new_solution)), 2)
        new_solution[job1], new_solution[job2] = new_solution[job2], new_solution[job1]
        
        new_makespan = calculate_makespan(new_solution)
        
        # 새로운 해의 적합도가 더 좋거나, 확률적으로 나쁜 해를 수용
        if new_makespan < current_makespan or random.random() < math.exp((current_makespan - new_makespan) / temp):
            current_solution = new_solution
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_solution = current_solution
                best_makespan = current_makespan
        
        temp *= 1 - cooling_rate
    
    return best_makespan, best_solution

def print_jobs_allocation(jobs):
    # 모든 작업을 검사하여 사용된 최대 기계 번호를 찾습니다.
    max_machine_number = max(machine for job in jobs for machine, _ in job)
    # 모든 기계에 대해 딕셔너리 키를 생성합니다.
    machine_allocations = {machine: [] for machine in range(1, max_machine_number + 1)}
    
    for job_id, job in enumerate(jobs, start=1):
        for machine, _ in job:
            if machine in machine_allocations:  # 이 조건은 사실상 항상 참이 됩니다.
                machine_allocations[machine].append(job_id)

    # for machine, assigned_jobs in machine_allocations.items():
    #     print(f"Machine {machine}: Jobs {assigned_jobs}")


# 문제 해결 및 출력
for i in range(1, 101):
    filename = f"problem_{i}.csv"
    makespan, jobs = simulated_annealing_solver(filename)
    print(f"Problem {i}: Total makespan = {makespan}")
    print_jobs_allocation(jobs)
