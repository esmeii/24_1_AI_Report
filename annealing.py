import random
import math
import time

# 코드 실행 전 시간 측정
start_time = time.time()

def initial_solution(schedule):
    # 초기 스케줄 생성
    return random.sample(schedule, len(schedule))

def get_makespan(schedule):
    # 각 작업의 소요 시간을 합산하여 makespan 계산
    total_time = 0
    for job in schedule:
        for task in job:
            total_time += task[1]  # task[1]은 각 작업의 소요 시간
    return total_time


def get_neighbour(schedule):
    # 이웃해(solution) 생성을 위한 두 작업의 위치를 무작위로 교환
    a, b = random.sample(range(len(schedule)), 2)
    new_schedule = schedule[:]
    new_schedule[a], new_schedule[b] = new_schedule[b], new_schedule[a]
    return new_schedule

def accept_probability(old_cost, new_cost, temperature):
    # 새로운 해가 더 나쁜 경우(즉, makespan이 더 큰 경우)에도 일정 확률로 선택
    if new_cost > old_cost:
        return 1.0
    else:
        return math.exp((old_cost - new_cost) / temperature)
import csv

def read_schedule_from_file(file_path):
    schedule = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # 각 줄의 양 끝에서 큰따옴표를 제거하고, 쉼표로 항목을 분리합니다.
            items = line.strip().strip('"').split('","')
            
            # 각 항목을 다시 쉼표로 분리하여 숫자 쌍으로 변환하고, 이를 정수로 변환합니다.
            row = [(int(pair.split(',')[0]), int(pair.split(',')[1])) for pair in items]
            schedule.append(row)
    
    return schedule

def simulated_annealing(filename):
    # 파일에서 작업 스케줄 읽기
    schedule = read_schedule_from_file(filename)
    
    temp = 20000
    final_temp = 10
    cooling_rate = 0.99
    
    current_solution = initial_solution(schedule)
    current_cost = get_makespan(current_solution)
    
    while temp > final_temp:
        new_solution = get_neighbour(current_solution)
        new_cost = get_makespan(new_solution)
        
        if accept_probability(current_cost, new_cost, temp) > random.random():
            current_solution = new_solution
            current_cost = new_cost
        
        temp *= cooling_rate
    
    return current_cost, current_solution


for i in range(1, 200):
    filename = f"problem_{i}.csv"
    makespan, jobs = simulated_annealing(filename)
    print(f"Problem {i}: makespan = {makespan}")
#    print("Job allocation:", jobs)

# 코드 실행 후 시간 측정
    end_time = time.time()

# 총 실행 시간을 계산하고 출력
    total_time = end_time - start_time
    print(f"총 실행 시간: {total_time}초")

