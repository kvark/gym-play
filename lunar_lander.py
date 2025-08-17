from enum import Enum
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1) # Output probabilities
        return action_probs

class Action(Enum):
    NOTHING = 0
    LEFT_ENGINE = 1
    MAIN_ENGINE = 2
    RIGHT_ENGINE = 3

STATE_DIM = 8
BATCH_SIZE = 100
MAX_STEPS = 200
ITERATIONS = 100000
PRESENT_STEP = 100
FUTURE_REWARD_WEIGHT = 0.80
DEMO_MODE = False

class Thread:
    def __init__(self, state):
        self.state = state
        self.done = False
        self.log_prob_actions = []
        self.rewards = []

present_env = gym.make("LunarLander-v3", render_mode="human")
train_env = gym.make("LunarLander-v3", render_mode=None)

policy_net = Policy(state_dim=STATE_DIM, action_dim=len(Action))
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.01)

for iter_id in range(ITERATIONS):
    observation, info = train_env.reset(seed = iter_id)
    assert len(observation) == STATE_DIM
    threads = [Thread(observation) for i in range(BATCH_SIZE)]
    for step_id in range(MAX_STEPS):
        has_updates = False
        for thread in threads:
            if thread.done:
                continue
            action_probabilities = policy_net(torch.from_numpy(thread.state).float())
            action = torch.multinomial(action_probabilities, 1).item()
            observation, reward, terminated, truncated, info = train_env.step(action)
            thread.state = observation
            thread.done = terminated or truncated
            log_prob = torch.log(action_probabilities[action])
            thread.log_prob_actions.append(log_prob)
            reward_t = torch.tensor(reward, dtype=torch.float32)
            thread.rewards.append(reward_t)
            #print("Action:", action, "LogProb", log_prob.item(), "Reward", reward)
            has_updates = True
        if not has_updates:
            break

    optimizer.zero_grad()
    for thread in threads:
        reward_current = 0
        rewards_cumulative = []
        for reward in reversed(thread.rewards):
            reward_current = reward + FUTURE_REWARD_WEIGHT * reward_current
            #print("Current", reward, "Cumulative", reward_current)
            rewards_cumulative.append(reward_current)
        rewards = torch.stack(list(reversed(rewards_cumulative)))
        log_probs = torch.stack(thread.log_prob_actions)
        loss = -(rewards * log_probs).sum()
        loss.backward()
    optimizer.step()

    if iter_id % PRESENT_STEP == 0:
        observation, info = present_env.reset(seed = iter_id)
        rewards = []
        actions_with_probabilities = []
        for step_id in range(MAX_STEPS):
            action_probabilities = policy_net(torch.from_numpy(observation).float())
            action = torch.multinomial(action_probabilities, 1).item()
            actions_with_probabilities.append((action, action_probabilities[action]))
            observation, reward, terminated, truncated, info = present_env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        print("Iteration:", iter_id, "reward:", sum(rewards))
        if DEMO_MODE:
            rewards_cumulative = []
            reward_current = 0
            for reward in reversed(rewards):
                reward_current = reward + FUTURE_REWARD_WEIGHT * reward_current
                rewards_cumulative.append(reward_current)
            rewards_cumulative = reversed(rewards_cumulative)
            observation, info = present_env.reset(seed = iter_id)
            for ((action, probability), (reward, reward_cum)) in zip(actions_with_probabilities, zip(rewards, rewards_cumulative)):
                observation, _, terminated, truncated, info = present_env.step(action)
                input(f"{Action(action)} ({probability}) -> {reward} ({reward_cum})")

train_env.close()
present_env.close()
