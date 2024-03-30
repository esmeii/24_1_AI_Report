import csv
import random

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
    # 간단한 방식으로 makespan을 계산합니다.
    # 이 부분은 JSP의 복잡성에 따라 매우 다르게 구현될 수 있습니다.
    max_time = 0
    for job in jobs:
        job_time = sum([time for _, time in job])
        if job_time > max_time:
            max_time = job_time
    return max_time

# 유전 알고리즘 구현 (매우 간소화된 버전)
def genetic_algorithm_solver(filename):
    jobs = read_problem_from_csv(filename)
    
    # 초기 개체군 생성 (여기서는 단순히 입력된 jobs를 사용)
    # 실제 구현에서는 다양한 순서의 jobs를 생성해야 합니다.
    initial_population = [jobs]  # 이 예제에서는 초기 개체군이 하나의 해만을 포함
    
    # 적합도 계산
    makespan = calculate_makespan(jobs)
    
    # 유전 알고리즘의 나머지 단계는 이 예제에서 생략됩니다.
    # 선택, 교차, 변이 등의 과정을 통해 최적화할 필요가 있습니다.

    return makespan, jobs

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
    makespan, jobs = genetic_algorithm_solver(filename)
    print(f"Problem {i}: Total makespan = {makespan}")
    print_jobs_allocation(jobs)
