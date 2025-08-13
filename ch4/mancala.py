!pip install torch numpy tqdm

import numpy as np

class MancalaEnv:
    """
    Defines the Mancala game environment for reinforcement learning.
    - Initializing and resetting the game board
    - Determining valid moves for each player
    - Applying moves and updating the game state
    - Calculating rewards and detecting game termination

    Returns current state, reward, and flags after each action.
    """
  
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [4]*6 + [0] + [4]*6 + [0]
        self.current_player = 0
        return self._get_state()

    def _get_state(self):
        return tuple(self.board + [self.current_player])

    def get_valid_actions(self):
        offset = 0 if self.current_player == 0 else 7
        return [i for i in range(offset, offset + 6) if self.board[i] > 0]

    def step(self, action):
        stones = self.board[action]
        self.board[action] = 0
        pos = action
        while stones > 0:
            pos = (pos + 1) % 14
            if (self.current_player == 0 and pos == 13) or (self.current_player == 1 and pos == 6):
                continue
            self.board[pos] += 1
            stones -= 1

        reward = 0
        done = False
        if self._game_over():
            done = True
            self._sweep_remaining_stones()
            p1, p2 = self.board[6], self.board[13]
            if p1 > p2:
                reward = 1 if self.current_player == 0 else -1
            elif p2 > p1:
                reward = -1 if self.current_player == 0 else 1
            else:
                reward = 0
        else:
            self.current_player = 1 - self.current_player

        return self._get_state(), reward, done

    def _sweep_remaining_stones(self):
        for i in range(6):
            self.board[6] += self.board[i]
            self.board[i] = 0
        for i in range(7, 13):
            self.board[13] += self.board[i]
            self.board[i] = 0

    def _game_over(self):
        return all(s == 0 for s in self.board[0:6]) or all(s == 0 for s in self.board[7:13])


# =========== put this code block in a new cell ===========

import torch
import torch.nn as nn
import torch.optim as optim
import random

class QNetwork(nn.Module):
    """
    Defines the neural network used to approximate Q-values.
    Takes game state as input and outputs a Q-value for each possible action.
    Allows the agent to learn and predict the best moves during training and play.
    """

    def __init__(self, input_size=15, output_size=14):  # 14 total actions
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# =========== put this code block in a new cell ===========
class DeepQLearningAgent:
    """
    Defines the reinforcement learning agent that uses a neural network (QNetwork)
    to approximate Q-values for each state-action pair.
    - Choosing actions using an epsilon-greedy policy
    - Updating the network based on experience (Q-learning rule)
    - Managing exploration vs. exploitation via epsilon decay

    """

    def __init__(self, lr=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor).squeeze()
                valid_q_values = [(a, q_values[a].item()) for a in valid_actions]
                max_q = max(valid_q_values, key=lambda x: x[1])[1]
                best_actions = [a for a, q in valid_q_values if q == max_q]
                return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_actions, done):
        self.model.train()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.model(state_tensor)
        target_q_values = q_values.clone().detach()

        if done or not next_actions:
            target_q_values[0, action] = reward
        else:
            with torch.no_grad():
                next_q_values = self.model(next_state_tensor)
                max_next_q = max([next_q_values[0, a].item() for a in next_actions])
                target_q_values[0, action] = reward + self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# =========== put this code block in a new cell ===========

from tqdm import trange

# Initialize mancala game
env = MancalaEnv()

# Reinforcement agent
agent = DeepQLearningAgent()

episodes = 10000

for ep in trange(episodes):
    state = env.reset()
    done = False
    # Plays the game until it ends
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions)
        next_state, reward, done = env.step(action)
        next_actions = env.get_valid_actions() if not done else []
        agent.learn(state, action, reward, next_state, next_actions, done)
        state = next_state
    agent.decay()

# =========== put this code block in a new cell ===========

# Evaluate the performance of the agent
def evaluate(agent, games=100):
    wins = 0
    for _ in range(games):
        # starts the game
        state = env.reset()
        done = False
         # Play until it ends
        while not done:
            valid_actions = env.get_valid_actions()
            if env.current_player == 0:
                action = agent.choose_action(state, valid_actions)
            else:
                action = random.choice(valid_actions)
            state, reward, done = env.step(action)
        # if agent wins, it gets a reward
        if reward == 1:
            wins += 1
    print(f"Win rate vs. random: {wins / games:.2f}")

# =========== put this code block in a new cell ===========

# Sets up mancala board
def render_board(state):
    p2 = state[7:13]
    p2_store = state[13]
    p1 = state[0:6]
    p1_store = state[6]

    print("     ", list(reversed(p2)))
    print(f"{p2_store} <--             --> {p1_store}")
    print("     ", list(p1))
    print(f"Current Player: {state[14]}")

# =========== put this code block in a new cell ===========

# Watch the agent learn and act 
def auto_play_verbose(agent, episodes=1):
    env = MancalaEnv()

    for ep in range(episodes):
        print(f"\nEpisode {ep+1}")
        state = env.reset()
        done = False
        render_board(state)

        while not done:
            valid_actions = env.get_valid_actions()
            player = env.current_player

            if player == 0:
                print("\nAgent's Turn")
                if random.random() < agent.epsilon:
                    mode = "Exploring"
                else:
                    mode = "Exploiting"

                action = agent.choose_action(state, valid_actions)

                # Show Q-values
                with torch.no_grad():
                    q_vals = agent.model(torch.FloatTensor(state).unsqueeze(0)).squeeze()
                    for i in valid_actions:
                        print(f"  Q[{i}] = {q_vals[i].item():.3f}")
                print(f"  ➤ {mode} | Action chosen: {action}")
            else:
                print("\nOpponent's Turn")
                action = random.choice(valid_actions)
                print(f"  ➤ Opponent chooses: {action}")

            next_state, reward, done = env.step(action)
            render_board(next_state)

            if player == 0 and not done:
                next_actions = env.get_valid_actions()
                agent.learn(state, action, reward, next_state, next_actions, done)

            state = next_state

        print("Game Over")
        if reward > 0:
            print("Agent Wins")
        elif reward < 0:
            print("Opponent Wins")
        else:
            print("Draw")

# =========== put this code block in a new cell ===========

auto_play_verbose(agent, episodes=1)


