import csv
import random
import math
import csv
import random
import time
# 코드 실행 전 시간 측정
start_time = time.time()

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
    
def rotate_sublist(lst, start, end, steps):
    sublist = lst[start:end]
    steps = steps % len(sublist)
    lst[start:end] = sublist[-steps:] + sublist[:-steps]

def shaking(current_solution, k, method):
    # 이전 이웃 생성 방법들의 구현은 유지
    new_solution = current_solution.copy()  # 현재 해 복사
    
    if method == "gap_adjust":
        # gap_adjust 방법 구현
        job_index = random.randint(0, len(new_solution) - 2)  # 마지막 작업은 제외
        gap = random.randint(1, 3)  # 조정할 간격 예시
        # 간단한 구현을 위해, 작업 간 간격을 조정하는 대신 작업 순서 변경으로 시뮬레이션
        new_solution.insert(job_index + gap if job_index + gap < len(new_solution) else len(new_solution) - 1, new_solution.pop(job_index))
        return new_solution
    
    elif method == "block_reverse":
        # block_reverse 방법 구현
        start_index = random.randint(0, len(new_solution) - k)  # 블록의 시작 위치
        block = new_solution[start_index:start_index + k]
        block.reverse()  # 블록 내 작업 순서 뒤집기
        new_solution = new_solution[:start_index] + block + new_solution[start_index + k:]
        return new_solution

    intensity = max(1, k // 2)  # 변형 강도를 조절하기 위한 변수, 최소값은 1

    if method == "exchange":
        for _ in range(intensity):
            job1, job2 = random.sample(range(len(new_solution)), 2)
            new_solution[job1], new_solution[job2] = new_solution[job2], new_solution[job1]
    
    elif method == "reverse":
        for _ in range(intensity):
            if len(new_solution) > 1:
                start = random.randint(0, len(new_solution) - 2)
                end = random.randint(start + 1, len(new_solution))
                new_solution[start:end] = reversed(new_solution[start:end])
    
    elif method == "insert":
        for _ in range(intensity):
            if len(new_solution) > 1:
                start = random.randint(0, len(new_solution) - 1)
                end = random.randint(0, len(new_solution) - 1)
                job = new_solution.pop(start)
                new_solution.insert(end, job)
    
    elif method == "rotate":
        for _ in range(intensity):
            if len(new_solution) > 1:
                start = random.randint(0, len(new_solution) - 2)
                end = random.randint(start + 1, len(new_solution))
                steps = random.randint(1, end - start)
                new_solution[start:end] = new_solution[start+steps:end] + new_solution[start:start+steps]

    return new_solution

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

def sum_nested(iterable):
    total = 0
    for item in iterable:
        if isinstance(item, (list, tuple)):
            total += sum_nested(item)  # 재귀 호출
        else:
            total += item
    return total

def generate_advanced_initial_solutions(jobs, method_count=5):
    """
    다양한 방법으로 초기 해를 생성하고, 가장 적은 make span을 보장하는 해들을 반환합니다.
    """
    neighborhood_functions = [exchange, reverse, insert, two_part_exchange, group_move]
    solutions = []

    # 원본 리스트를 기반으로 한 해
    solutions.append((jobs[:], calculate_makespan(jobs)))

    # 각 함수를 사용하여 해 생성
    for func in neighborhood_functions:
        new_solution = func(jobs[:], 1)  # k=1로 설정하여 각 함수를 한 번씩 적용
        solutions.append((new_solution, calculate_makespan(new_solution)))

    # 모든 해를 make span에 따라 정렬
    solutions.sort(key=lambda x: x[1])

    # 가장 좋은 해들을 선택
    selected_solutions = [solution for solution, _ in solutions[:method_count]]

    return selected_solutions

def variable_neighborhood_search_less_optimized(filename):
    start_time = time.time()
    jobs = read_problem_from_csv(filename)
    initial_solutions = generate_advanced_initial_solutions(jobs)
    best_solution = None
    best_makespan = float('inf')

    # 초기 해답 집합에서 무작위로 하나의 해만 선택
    initial_solution = random.choice(initial_solutions)
    current_solution = initial_solution
    current_makespan = calculate_makespan(current_solution)

    if current_makespan < best_makespan:
        best_solution = current_solution
        best_makespan = current_makespan

    k_max = int(math.sqrt(len(jobs))) // 4  # k_max를 더 작게 설정
    k = 1
    methods = ["exchange", "reverse"]  # 사용 가능한 탐색 방법을 줄임
    iteration = 0
    stuck_counter = 0

    while time.time() - start_time < 1:  # 실행 시간을 30초로 제한
        method = random.choice(methods)  # 탐색 방법을 무작위로 선택
        new_solution = shaking(current_solution, k, method)
        
        new_makespan = calculate_makespan(new_solution)
        if new_makespan < current_makespan:
            current_solution = new_solution
            current_makespan = new_makespan
            if new_makespan < best_makespan:
                best_solution = new_solution
                best_makespan = new_makespan
            k = 1  # k 값을 초기화
            stuck_counter = 0
        else:
            stuck_counter += 1
            if stuck_counter >= 1:  # stuck_counter 조건을 더 강화
                k += 1  # k 값을 증가
                stuck_counter = 0
        if k > k_max:
            k = 1

        iteration += 1
        if iteration >= 3:  # 최대 반복 횟수를 줄임
            break

    return best_makespan, best_solution

for i in range(1, 201):
    filename = f"problem_{i}.csv"
    makespan, _ = variable_neighborhood_search_less_optimized(filename)
    print(f"Problem {i}: makespan = {makespan}")
# # 코드 실행 후 시간 측정
#     end_time = time.time()

# # 총 실행 시간을 계산하고 출력
#     total_time = end_time - start_time
#     print(f"총 실행 시간: {total_time}초")

