import csv
import random
import math
import csv
import random
import time

def exchange(jobs, k):
    """
    무작위로 두 작업을 선택하여 위치를 교환합니다.
    """
    new_jobs = jobs[:]
    for _ in range(k):
        i, j = random.sample(range(len(jobs)), 2)
        new_jobs[i], new_jobs[j] = new_jobs[j], new_jobs[i]
    return new_jobs

def reverse(jobs, k):
    """
    무작위로 선택한 부분 리스트를 역순으로 합니다.
    """
    new_jobs = jobs[:]
    if len(jobs) > 1:
        start = random.randint(0, len(jobs) - 2)
        end = random.randint(start + 1, len(jobs))
        new_jobs[start:end] = reversed(new_jobs[start:end])
    return new_jobs

def insert(jobs, k):
    """
    무작위로 선택한 작업을 다른 위치로 이동합니다.
    """
    new_jobs = jobs[:]
    for _ in range(k):
        i = random.randint(0, len(jobs) - 1)
        job = new_jobs.pop(i)
        new_pos = random.randint(0, len(new_jobs))
        new_jobs.insert(new_pos, job)
    return new_jobs

def two_part_exchange(jobs, k):
    """
    무작위로 선택한 두 부분의 작업 순서를 교환합니다.
    """
    new_jobs = jobs[:]
    if len(jobs) > 2 and k > 1:
        part1_start = random.randint(0, len(jobs) - k - 1)
        part2_start = random.randint(part1_start + k, len(jobs) - 1)
        new_jobs[part1_start:part1_start+k], new_jobs[part2_start:part2_start+k] = new_jobs[part2_start:part2_start+k], new_jobs[part1_start:part1_start+k]
    return new_jobs

def group_move(jobs, k):
    """
    무작위로 선택한 작업 그룹을 다른 위치로 이동합니다.
    """
    new_jobs = jobs[:]
    if len(jobs) > k:
        start = random.randint(0, len(jobs) - k)
        group = new_jobs[start:start+k]
        del new_jobs[start:start+k]
        new_pos = random.randint(0, len(new_jobs))
        for i in range(k):
            new_jobs.insert(new_pos + i, group[i])
    return new_jobs


def reverse_sublist(lst, start, end):
    lst[start:end] = lst[start:end][::-1]

def insert_sublist(lst, start, end):
    element = lst.pop(start)
    lst.insert(end, element)

def shaking(jobs, k, method="exchange"):
    new_jobs = [list(job) for job in jobs]
    if method == "exchange":
        for _ in range(k):
            job1, job2 = random.sample(range(len(new_jobs)), 2)
            new_jobs[job1], new_jobs[job2] = new_jobs[job2], new_jobs[job1]
    elif method == "reverse":
        if len(new_jobs) > 1:
            start = random.randint(0, len(new_jobs) - 2)
            end = random.randint(start + 1, len(new_jobs))
            reverse_sublist(new_jobs, start, end)
    elif method == "insert":
        if len(new_jobs) > 1:
            start = random.randint(0, len(new_jobs) - 1)
            end = random.randint(0, len(new_jobs) - 1)
            insert_sublist(new_jobs, start, end)
    return new_jobs

def read_problem_from_csv(filename):
    jobs = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            job = [(int(machine), int(time)) for machine, time in (task.split(',') for task in row)]
            jobs.append(job)
    return jobs

# def calculate_makespan(jobs):
#     max_time = 0
#     for job in jobs:
#         job_time = sum([time for _, time in job])
#         if job_time > max_time:
#             max_time = job_time
#     return max_time

def calculate_makespan(jobs):
    """
    작업 스케줄의 총 처리 시간(makespan)을 계산합니다.
    """
    machine_times = {}
    for job in jobs:
        current_start_time = 0
        for machine, time in job:
            if machine not in machine_times:
                machine_times[machine] = 0
            start_time = max(current_start_time, machine_times[machine])
            machine_times[machine] = start_time + time
            current_start_time = machine_times[machine]
    return max(machine_times.values())

def generate_initial_solutions(jobs, method_count=5):
    """
    다양한 방법으로 초기 해를 생성합니다.
    """
    solutions = []

    # 원본 리스트를 기반으로 한 해
    solutions.append(jobs[:])

    # 무작위 해
    for _ in range(method_count-1):
        new_jobs = jobs[:]
        random.shuffle(new_jobs)
        solutions.append(new_jobs)
    
    return solutions

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

def variable_neighborhood_search_improved(filename):
    start_time = time.time()
    jobs = read_problem_from_csv(filename)
    initial_solutions = generate_initial_solutions(jobs)
    best_solution = None
    best_makespan = float('inf')

    for initial_solution in initial_solutions:
        current_solution = initial_solution
        current_makespan = calculate_makespan(current_solution)

        if current_makespan < best_makespan:
            best_solution = current_solution
            best_makespan = current_makespan

        k_max = int(math.sqrt(len(jobs)))  # 동적 k_max 설정
        k = 1
        methods = ["exchange", "reverse", "insert", "two_part_exchange", "group_move"]  # 추가 탐색 방법
        iteration = 0
        stuck_counter = 0

        while time.time() - start_time < 60:  # 1분 이내에 실행되도록 설정
            for method in methods:
                new_solution = globals()[method](current_solution, k)
                new_makespan = calculate_makespan(new_solution)
                if new_makespan < current_makespan:
                    current_solution = new_solution
                    current_makespan = new_makespan
                    if new_makespan < best_makespan:
                        best_solution = new_solution
                        best_makespan = new_makespan
                    k = 1  # k 값을 초기화
                    break
                else:
                    stuck_counter += 1
                    if stuck_counter >= len(methods):
                        k += 1  # k 값을 증가
                        stuck_counter = 0
                if k > k_max:
                    k = 1

            iteration += 1
            if iteration >= 1000:  # 최대 반복 횟수 제한
                break

    return best_makespan, best_solution

def print_jobs_allocation(jobs):
    max_machine_number = max(machine for job in jobs for machine, _ in job)
    machine_allocations = {machine: [] for machine in range(1, max_machine_number + 1)}
    
    for job_id, job in enumerate(jobs, start=1):
        for machine, _ in job:
            machine_allocations[machine].append(job_id)

for i in range(1, 101):
    filename = f"problem_{i}.csv"
    makespan, _ = variable_neighborhood_search_improved(filename)
    print(f"문제 {i}: 총 처리 시간 = {makespan}")


