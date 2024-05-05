import csv
import random
import math
import time
import numpy as np

# 코드 실행 전 시간 측정
start_time = time.time()

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

def swap_jobs(solution, index1, index2):
    solution[index1], solution[index2] = solution[index2], solution[index1]

def move_shortest_task_to_front(job):
    if len(job) > 1:
        shortest_task_index = min(range(len(job)), key=lambda index: job[index][1])
        # 가장 짧은 작업을 맨 앞으로 이동
        job.insert(0, job.pop(shortest_task_index))

def optimize_critical_path(solution):
    critical_jobs = sorted(solution, key=lambda job: sum(time for _, time in job), reverse=True)
    if len(critical_jobs) > 2:
        swap_jobs(critical_jobs, 0, 1)  # 가장 긴 두 작업의 위치를 바꿈
    return critical_jobs

def optimize_machine_distribution_with_MRPT(solution):
    machine_loads = {}
    for job in solution:
        for machine, time in job:
            if machine not in machine_loads:
                machine_loads[machine] = 0
            machine_loads[machine] += time
    # 가장 바쁜 기계 찾기
    busiest_machine = max(machine_loads, key=machine_loads.get)
    # 해당 기계를 사용하는 작업들 중 남은 처리 시간이 가장 긴 작업을 찾아서 맨 앞으로 이동
    jobs_using_busiest_machine = [(job_index, job) for job_index, job in enumerate(solution) if any(machine == busiest_machine for machine, _ in job)]
    
    if len(jobs_using_busiest_machine) > 1:
        # 남은 처리 시간 계산
        remaining_times = [(job_index, sum(time for _, time in job)) for job_index, job in jobs_using_busiest_machine]
        # 남은 처리 시간이 가장 긴 작업을 찾음
        longest_remaining_job_index = max(remaining_times, key=lambda x: x[1])[0]
        # 해당 작업을 맨 앞으로 이동
        solution.insert(0, solution.pop(longest_remaining_job_index))
        
    return solution
# 기계별 작업 완료 시간을 업데이트하는 함수 추가
def update_machine_completion_times(machine_completion_times, job, machine_allocations):
    for machine, time in job:
        if machine not in machine_completion_times:
            machine_completion_times[machine] = 0
        machine_completion_times[machine] += time
        machine_allocations[machine].append(machine_completion_times[machine])
    return machine_completion_times, machine_allocations

def allocate_jobs_with_ECT(solution, machine_completion_times):
    machine_allocations = {machine: [] for machine in machine_completion_times.keys()}
    allocated_jobs = []

    for job in solution:
        allocated_job = []
        for task in job:
            # 각 태스크를 처리할 수 있는 가장 빠른 기계를 찾습니다.
            best_machine = min(machine_completion_times, key=lambda x: machine_completion_times[x])
            # 해당 기계의 완료 시간을 업데이트합니다.
            machine_completion_times[best_machine] += task[1]
            # 할당된 태스크 정보를 업데이트합니다. (기계 번호, 처리 시간)
            allocated_job.append((best_machine, task[1]))
            # 기계별 할당된 작업 시간을 업데이트합니다.
            machine_allocations[best_machine].append(machine_completion_times[best_machine])
        # 전체 작업에 대한 할당 결과를 업데이트합니다.
        allocated_jobs.append(allocated_job)
    
    # 수정된 함수에서는 각 작업에 대한 할당된 결과를 반환합니다.
    return allocated_jobs, machine_allocations

def swap_jobs(solution, job_index1, job_index2):
    # 두 작업 위치를 교환하는 로직을 구현합니다.
    temp = solution[job_index1]
    solution[job_index1] = solution[job_index2]
    solution[job_index2] = temp

def simulated_annealing_solver(filename):
    jobs = read_problem_from_csv(filename)
    current_solution = jobs
    current_makespan = calculate_makespan(current_solution)
    best_solution = current_solution
    best_makespan = current_makespan
    
    temp = 10000  # 초기 온도
    final_temp = 0.1  # 최종 온도
    cooling_rate = 0.03  # 초기 냉각률
    no_improvement_steps = 0  # 개선되지 않은 단계 수
    
    while temp > final_temp:
        new_solution = [list(job) for job in current_solution]
        
        # 랜덤으로 작업을 선택하여 위치를 교환합니다.
        job_index1, job_index2 = random.sample(range(len(new_solution)), 2)
        swap_jobs(new_solution, job_index1, job_index2)
        
        new_makespan = calculate_makespan(new_solution)
        
        if new_makespan < current_makespan or math.exp((current_makespan - new_makespan) / temp) > random.random():
            current_solution = new_solution
            current_makespan = new_makespan
            no_improvement_steps = 0  # 개선되었으므로 카운터를 초기화합니다.
            
            if new_makespan < best_makespan:
                best_solution = current_solution
                best_makespan = new_makespan
        else:
            no_improvement_steps += 1  # 개선되지 않았으므로 카운터를 증가시킵니다.
        
        # 적응적 냉각률 조정
        if no_improvement_steps > 10:  # 해가 10단계 동안 개선되지 않은 경우
            cooling_rate = min(0.05, cooling_rate + 0.01)  # 냉각률을 증가
            no_improvement_steps = 0  # 카운터 초기화
        else:
            cooling_rate = max(0.01, cooling_rate - 0.001)  # 해가 개선될 때마다 냉각률을 약간 감소
        
        temp *= 1 - cooling_rate
    
    return best_makespan, best_solution

def print_jobs_allocation(jobs):
    max_machine_number = max(machine for job in jobs for machine, _ in job)
    machine_allocations = {machine: [] for machine in range(1, max_machine_number + 1)}
    
    for job_id, job in enumerate(jobs, start=1):
        for machine, _ in job:
            machine_allocations[machine].append(job_id)

# 예시 사용
for i in range(302, 401):
    filename = f"problem_{i}.csv"
    makespan, jobs = simulated_annealing_solver(filename)
    print(f"Problem {i}: makespan = {makespan}")
    print_jobs_allocation(jobs)
#     # 코드 실행 후 시간 측정
#     end_time = time.time()

# # 총 실행 시간을 계산하고 출력
#     total_time = end_time - start_time
#     print(f"총 실행 시간: {total_time}초")
