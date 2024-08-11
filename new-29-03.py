import random
import argparse

import numpy as np
import gymnasium as gym


parser = argparse.ArgumentParser()

parser.add_argument(
    'epsilon',
    help='Epsilon value should be between 0 and 1',
    type=float,
    default=0.1
)

args = parser.parse_args()

env = gym.make("Taxi-v3", render_mode='ansi')

q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = args.epsilon

for epoch in range(100000):
    state = env.reset()[0]

    while True:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, terminated, truncated, info = env.step(action)

        # in case of termination or truncation we are going to step this epoch
        # and reset the environment
        if terminated or truncated:
            break

        q_table[state][action] = (
            (1 - alpha) * q_table[state][action] +
            alpha * (reward + gamma * np.max(q_table[next_state]))
        )

        state = next_state

    if epoch % 10000 == 0:
        print(f'Epoch - {epoch}')

print("Training finished.\n")
print(q_table)
