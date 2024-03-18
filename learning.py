import numpy as np
import csv

# Q-learning parameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

# Environment parameters
num_jobs = 36
num_machines = 2
num_actions = num_jobs * num_machines  # Number of possible actions

# Initialize Q-table
Q = np.zeros((num_jobs, num_actions))

# Function to select an action
def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)  # Exploration: choose random action
    else:
        return np.argmax(Q[state])  # Exploitation: choose best action

# Function to update Q-table
def update_Q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state])
    Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

# Function to simulate an episode
def simulate_episode(jobs):
    state = 0  # Start state
    total_completion_time = 0
    
    for job in jobs:
        for task in job:
            action = select_action(state)
            machine_id, processing_time = task
            total_completion_time += processing_time
            state = (state + 1) % num_jobs  # Ensure state stays within valid range
            update_Q(state-1, action, -processing_time, state)  # Update Q-table
    return total_completion_time

# Solve problems from 1 to 100
for problem_number in range(1, 101):
    # Load problem data
    filename = f"problem_{problem_number}.csv"
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        jobs = [[tuple(map(int, task.split(','))) for task in row] for row in reader]

    # Solve problem using Q-learning
    print(f"Solving problem {problem_number}...")
    completion_time = simulate_episode(jobs)

    # Print results
    print(f"Best schedule for problem {problem_number}:")
    for i, job in enumerate(jobs, start=1):
        print(f"Job {i}: {job}")
    print(f"Completion time: {completion_time}")


# Load problem data
problem_number = 100
filename = f"problem_{problem_number}.csv"
with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    jobs = [[tuple(map(int, task.split(','))) for task in row] for row in reader]

# Solve problem using Q-learning
print(f"Solving problem {problem_number}...")
completion_time = simulate_episode(jobs)

# Print results
print(f"Best schedule for problem {problem_number}:")
for i, job in enumerate(jobs, start=1):
    print(f"Job {i}: {job}")
print(f"Completion time: {completion_time}")
