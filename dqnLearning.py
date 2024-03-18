import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

# Define the neural network architecture
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate
batch_size = 64
num_epochs = 50

# Environment parameters
num_jobs = 36
num_machines = 2
num_actions = num_jobs * num_machines  # Number of possible actions

# Initialize DQN
dqn = DQN(input_size=num_jobs * num_machines, output_size=num_actions)
optimizer = optim.Adam(dqn.parameters(), lr=alpha)
criterion = nn.MSELoss()

# Function to select an action
def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)  # Exploration: choose random action
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = dqn(state)
            return torch.argmax(q_values).item()  # Exploitation: choose best action

# Function to update DQN
def update_dqn(states, actions, rewards, next_states):
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)

    q_values = dqn(states)
    q_values_next = dqn(next_states).detach()
    q_values_targets = q_values.clone()

    for i in range(len(actions)):
        q_values_targets[i][actions[i]] = rewards[i] + gamma * torch.max(q_values_next[i])

    loss = criterion(q_values, q_values_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Function to simulate an episode
def simulate_episode(jobs):
    state = [0] * (num_jobs * num_machines)  # Start state
    total_completion_time = 0
    
    for job in jobs:
        for task in job:
            action = select_action(state)
            machine_id, processing_time = task
            total_completion_time += processing_time
            state[action] += processing_time  # Update state
            update_dqn([state], [action], [-processing_time], [state])  # Update DQN
    return total_completion_time

# Solve problems from 1 to 100
for problem_number in range(1, 101):
    # Load problem data
    filename = f"problem_{problem_number}.csv"
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        jobs = [[tuple(map(int, task.split(','))) for task in row] for row in reader]

    # Solve problem using DQN
    print(f"Solving problem {problem_number}...")
    completion_time = simulate_episode(jobs)

    # Print results
    print(f"Best schedule for problem {problem_number}:")
    print(f"Completion time: {completion_time}")
