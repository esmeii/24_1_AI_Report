import numpy as np
import random
import csv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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
        # Pad state to ensure consistent size
        state += [0] * ((self.num_machines * 2 + 2) - len(state))
        return np.array(state)

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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        # 수정된 DQNAgent 클래스의 act 메서드
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array(state))  # state를 numpy 배열로 변환
        return np.argmax(act_values[0])
    
    # 수정된 DQNAgent 클래스의 replay 메서드
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(np.array(next_state))[0]))  # state를 numpy 배열로 변환
            target_f = self.model.predict(np.array(state))
            target_f[0][action] = target
            self.model.fit(np.array(state), target_f, epochs=1, verbose=0)


# Load JSP problems from CSV files
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


# Train DQN agent on JSP problems
def train_agent(problems, num_episodes=10000, batch_size=32, replay_start_size=100):
    num_states = max(len(job) for problem in problems for job in problem) * 2 + 2
    num_actions = max(max(task[0] for job in problem for task in job) for problem in problems) + 1
    agent = DQNAgent(num_states, num_actions)

    num_training_problems = 900
    num_testing_problems = 100

    training_problems = problems[:num_training_problems]
    testing_problems = problems[num_training_problems:]

    for episode in range(num_episodes):
        total_reward = 0
        total_makespan = 0
        for problem in training_problems:
            env = JobShopSchedulingEnv(problem)
            state = env.reset()
            state = np.reshape(state, [1, num_states])
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, num_states])
                total_reward += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
            total_makespan += env.calculate_makespan()
        
        # Start replay when memory size reaches replay_start_size
        if len(agent.memory) >= replay_start_size:
            for _ in range(10):  # Replay 10 times
                agent.replay(batch_size)

        if (episode + 1) % 100 == 0:
            print(f"Training - Episode {episode + 1}, Total Reward: {total_reward}, Total Makespan: {total_makespan}")

    # Evaluate on test problems
    total_reward_test = 0
    total_makespan_test = 0
    for i, problem in enumerate(testing_problems, 1):
        env = JobShopSchedulingEnv(problem)
        state = env.reset()
        state = np.reshape(state, [1, num_states])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, num_states])
            total_reward_test += reward
            state = next_state
        total_makespan_test += env.calculate_makespan()
        print(f"Testing - Problem {i}/{num_testing_problems}, Total Reward: {total_reward_test}, Total Makespan: {total_makespan_test}")

# Example usage
if __name__ == "__main__":
    file_paths = [f"problem_{i}.csv" for i in range(1, 1001)]  # Path to JSP problem
    problems = load_problems_from_csv(file_paths)
    train_agent(problems)
