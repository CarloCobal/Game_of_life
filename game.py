#Mar 27, QCB: See Bottom for error line about dimension mismatch.

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy
import random
from multiprocessing import Pool, cpu_count
# from MicroMacroImplementation import game_of_life_macro
# from game_of_life_macro import game_of_life_macro



def game_of_life_micro(board):
    new_board = deepcopy(board)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            total = np.sum(board[max(i - 1, 0):min(i + 2, board.shape[0]),
                           max(j - 1, 0):min(j + 2, board.shape[1])]) - board[i, j]
            if board[i, j] == 1:
                new_board[i, j] = 1 if 2 <= total <= 3 else 0
            else:
                new_board[i, j] = 1 if total == 3 else 0
    return new_board


def game_of_life_macro(board, micro_size):
    macro_size = board.shape[0] // micro_size
    new_board = np.zeros_like(board)
    for i in range(macro_size):
        for j in range(macro_size):
            micro_board = board[i*micro_size:(i+1) *
                                micro_size, j*micro_size:(j+1)*micro_size]
            new_micro_board = game_of_life_micro(micro_board)
            new_board[i*micro_size:(i+1)*micro_size, j *
                      micro_size:(j+1)*micro_size] = new_micro_board
    return new_board

micro_size= 5


# Fitness function
def fitness(board, negative_noise=0.5):
    ticks = 0
    while np.sum(board) > 0:
        # board = game_of_life(board)
        # board = game_of_life_macro(board, micro_size)
        noise = np.random.rand(board.shape[0], board.shape[1]) < negative_noise
        board = np.logical_and(board, np.logical_not(noise)).astype(int)
        ticks += 1
    return ticks

# Evolutionary algorithm


def evolve(templates, mutation_rate=0.05):
    new_templates = []
    for template in templates:
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                if random.random() < mutation_rate:
                    template[i, j] = 1 - template[i, j]
        new_templates.append(template)
    return new_templates


def evaluate_fitness_parallel(templates, n_processes=None):
    if n_processes is None:
        n_processes = cpu_count()

    with Pool(processes=n_processes) as pool:
        fit_scores = pool.map(fitness, templates)

    return fit_scores


# Parameters
num_templates = 2
grid_size = 100
num_generations = 10  # ten mill generations.
mutation_rate = 0.05
max_score = -9999

# Initialize templates
templates = [np.random.randint(2, size=(grid_size, grid_size))
             for _ in range(num_templates)]
# Run multiple trials

for gen in range(num_generations):
    # fit_scores = evaluate_fitness_parallel(templates)
    fit_scores = [fitness(template) for template in templates]
    print(f"Generation {gen + 1} - Max Fitness: {max(fit_scores)}")
    for i in fit_scores:
        if max(fit_scores) > max_score:
            max_score = max(fit_scores)
    print(f"Max Score: {max_score}")

    # Select top performers
    sorted_indices = sorted(range(
        len(fit_scores)), key=lambda i: fit_scores[i], reverse=True)[:num_templates//2]
    top_templates = [templates[i] for i in sorted_indices]

    # Reproduce and mutate
    new_templates = top_templates * 2
    new_templates = evolve(new_templates, mutation_rate=mutation_rate)

    templates = new_templates

# Visualize the best template
best_template = templates[sorted_indices[0]]
fig, ax = plt.subplots()
ax.imshow(best_template, cmap="binary")
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_title("Best Template")
# # plt.show()

# save as an image:
plt.savefig('best_template.png')


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, learning_rate=0.001, memory_size=10000):
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()


    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states_tensor = torch.tensor(
            # states, dtype=torch.float32).to(self.device)
        states_tensor = torch.tensor(
            [s.flatten() for s in states], dtype=torch.float32).to(self.device)


        next_states_tensor = torch.tensor(
        [ns.flatten() for ns in next_states], dtype=torch.float32).to(self.device)

        actions_tensor = torch.tensor(
            actions, dtype=torch.int64).to(self.device)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32).to(self.device)
        # next_states_tensor = torch.tensor(
            # next_states, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)

        q_values = self.model(states_tensor).gather(
            1, actions_tensor.unsqueeze(1)).squeeze()
        next_q_values = self.model(next_states_tensor).max(1)[0].detach()
        target_q_values = rewards_tensor + \
            self.gamma * next_q_values * (~dones_tensor)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def detect_nonlinear_jumps(scores, threshold_factor=2.0):
    average_score = sum(scores) / len(scores)
    threshold = average_score * threshold_factor
    jumps = []

    for i, score in enumerate(scores):
        if score > threshold:
            jumps.append((i, score))

    return jumps

# Parameters
grid_size = 10
num_episodes = 10000
max_timesteps = 100
grid_size = micro_size
# state_size = grid_size * grid_size
state_size = micro_size * micro_size

action_size = grid_size

# Helper functions


# def encode_state_action(state, action):
#     state_flat = state.flatten()
#     action_onehot = np.zeros(grid_size * grid_size)
#     action_onehot[action] = 1
#     return np.concatenate((state_flat, action_onehot))
def encode_state(state):
    state_flat = state.flatten()
    return state_flat

def decode_action(action_idx):
    row = action_idx // grid_size
    col = action_idx % grid_size
    return row, col


# Training loop
episode_rewards = []
best_board = None
best_reward = -1

# agent = DQNAgent(micro_size * micro_size + action_size, action_size)
agent = DQNAgent(state_size, action_size)

for episode in range(num_episodes):
    # Initialize the environment
    board = np.random.randint(2, size=(grid_size, grid_size))

    total_reward = 0
    for timestep in range(max_timesteps):
        state = deepcopy(board)

        # Select an action
        # Action can be any valid index as it will be replaced by the agent's action
        # encoded_state = encode_state_action(state, 0)
        # action_idx = agent.choose_action(encoded_state)
        # action_idx = agent.choose_action(encoded_state[:-action_size])
        encoded_state = encode_state(state)
        action_idx = agent.choose_action(encoded_state)
        row, col = decode_action(action_idx)

        # Execute the action
        board[row, col] = 1 - board[row, col]
        board = game_of_life_macro(board, micro_size)

        # Compute the reward
        reward = fitness(board)
        total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_board = deepcopy(board)
        # Observe the next state
        next_state = deepcopy(board)

        # Store experience in the replay buffer
        done = (timestep == max_timesteps - 1)
        # Action can be any valid index as it will be replaced by the agent's action
        # encoded_next_state = encode_state_action(next_state, 0)
        # agent.remember(encoded_state, action_idx,
                    #    reward, encoded_next_state, done)
        encoded_next_state = encode_state(next_state)
        agent.remember(encoded_state, action_idx, reward, encoded_next_state, done)
        # Learn from the experiences
        agent.learn()

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

nonlinear_jumps = detect_nonlinear_jumps(episode_rewards)


if nonlinear_jumps:
    print("Non-linear jumps detected:")
    for idx, score in nonlinear_jumps:
        print(f"Episode {idx + 1}: Total Reward = {score}")
else:
    print("No non-linear jumps detected.")

fig, ax = plt.subplots()
ax.imshow(best_board, cmap="binary")
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.title(f"Best Board with Total Reward: {best_reward}")
plt.show()
