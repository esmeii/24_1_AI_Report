import csv
import random
import time
import copy

# 코드 실행 전 시간 측정
start_time = time.time()

def tournament_selection(population, scores, k=3):
    # 토너먼트 선택
    selection_ix = random.randint(0, len(population)-1)
    for ix in random.sample(range(len(population)), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

def crossover(p1, p2, crossover_rate=0.9):
    # 단일 지점 교차
    if random.random() < crossover_rate:
        cpoint = random.randint(1, len(p1)-2)
        child1 = p1[:cpoint] + p2[cpoint:]
        child2 = p2[:cpoint] + p1[cpoint:]
        return [child1, child2]
    else:
        return [p1, p2]

def mutation(individual, mutation_rate=0.1):
    # 변이: 작업 순서를 무작위로 변경
    if random.random() < mutation_rate:
        ix1, ix2 = random.sample(range(len(individual)), 2)
        individual[ix1], individual[ix2] = individual[ix2], individual[ix1]
    return individual

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

import copy
import random

def genetic_algorithm_solver(filename):
    jobs = read_problem_from_csv(filename)
    population_size = 50
    initial_population = [copy.deepcopy(jobs) for _ in range(population_size)]
    for individual in initial_population:
        random.shuffle(individual)

    n_generations = 100
    for generation in range(n_generations):
        scores = [calculate_makespan(individual) for individual in initial_population]
        ranked = sorted(zip(scores, initial_population), key=lambda x: x[0])
        selected = [x[1] for x in ranked[:population_size//2]]

        children = []
        while len(children) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            for child in crossover(parent1, parent2):
                child = mutation(child)
                children.append(child)
        initial_population = children

    scores = [calculate_makespan(individual) for individual in initial_population]
    best_score = min(scores)
    best_index = scores.index(best_score)
    best_solution = initial_population[best_index]

    return best_score, best_solution


def print_jobs_allocation(jobs):
    # 모든 작업을 검사하여 사용된 최대 기계 번호를 찾습니다.
    max_machine_number = max(machine for job in jobs for machine, _ in job)
    # 모든 기계에 대해 딕셔너리 키를 생성합니다.
    machine_allocations = {machine: [] for machine in range(1, max_machine_number + 1)}
    
    for job_id, job in enumerate(jobs, start=1):
        for machine, _ in job:
            if machine in machine_allocations:  # 이 조건은 사실상 항상 참이 됩니다.
                machine_allocations[machine].append(job_id)

    # 문제 해결 및 출력
for i in range(1, 200):
    filename = f"problem_{i}.csv"
    makespan, jobs = genetic_algorithm_solver(filename)
    print(f"Problem {i}: makespan = {makespan}")
    print_jobs_allocation(jobs)
    # 코드 실행 후 시간 측정
    end_time = time.time()

# 총 실행 시간을 계산하고 출력
    total_time = end_time - start_time
    print(f"총 실행 시간: {total_time}초")
