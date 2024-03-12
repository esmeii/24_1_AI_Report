import numpy as np
import random
import csv

class JobShopSchedulingEnv:
    def __init__(self, jobs):
        self.jobs = jobs
        self.num_jobs = len(jobs)
        self.num_machines = max(task[0] for job in jobs for task in job)
        self.machine_times = [0] * self.num_machines
        self.current_job = 0
        self.current_task = 0
        self.done = False

    def reset(self):
        self.machine_times = [0] * self.num_machines
        self.current_job = 0
        self.current_task = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        state = [self.machine_times[machine] for machine in range(self.num_machines)]
        state.extend([self.current_job, self.current_task])
        return tuple(state)

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Please reset the environment.")

        machine, processing_time = self.jobs[self.current_job][self.current_task]
        self.machine_times[machine - 1] += processing_time

        self.current_task += 1
        if self.current_task >= len(self.jobs[self.current_job]):
            self.current_job += 1
            self.current_task = 0
            if self.current_job >= self.num_jobs:
                self.done = True

        next_state = self.get_state()
        reward = -1  # Constant penalty for each step

        if self.done:
            reward += self.calculate_makespan() * -0.1  # Additional penalty based on makespan

        return next_state, reward, self.done

    def calculate_makespan(self):
        return max(self.machine_times)



class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[self.state_to_index(state)])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[self.state_to_index(next_state)])
        old_value = self.q_table[self.state_to_index(state), action]
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * self.q_table[self.state_to_index(next_state), best_next_action] - old_value)
        self.q_table[self.state_to_index(state), action] = new_value

    def decay_exploration_rate(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)

    def state_to_index(self, state):
        return hash(state) % self.num_states

    
# Load JSP problems from CSV files
def load_problems_from_csv(file_paths):
    problems = []
    for file_path in file_paths:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            problem = []
            for row in reader:
                job = []
                for task in row:
                    machine, processing_time = map(int, task.split(','))
                    job.append((machine, processing_time))
                problem.append(job)
            problems.append(problem)
    return problems
 
# Train Q-learning agent on JSP problems
def train_agent(problems, num_episodes=1000):
    num_states = max(len(job) for problem in problems for job in problem) * 2 + 2
    num_actions = max(max(task[0] for job in problem for task in job) for problem in problems) + 1
    agent = QLearningAgent(num_states, num_actions)

    for episode in range(num_episodes):
        total_reward = 0
        total_makespan = 0  # 총 makespan을 기록하기 위한 변수 추가
        for problem in problems:
            env = JobShopSchedulingEnv(problem)
            state = env.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                agent.update_q_table(agent.state_to_index(state), action, reward, agent.state_to_index(next_state))
                state = next_state
            total_makespan += env.calculate_makespan()  # 총 makespan 누적
            agent.decay_exploration_rate()
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Total Makespan: {total_makespan}")

    return agent



# Example usage
if __name__ == "__main__":
    file_paths = [f"problem_{i}.csv" for i in range(1, 51)]  # Path to JSP problem CSV files
    problems = load_problems_from_csv(file_paths)
    agent = train_agent(problems)
