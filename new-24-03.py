import random

import numpy as np
import gymnasium as gym


env = gym.make("Taxi-v3", render_mode='ansi')

q_table = np.zeros((env.observation_space.n, env.action_space.n))

epsilon = 0.5
alpha = 0.1
gamma = 0.6


for epoch in range(10000):
    print(f'Epoch - {epoch}')

    state = env.reset()

    penalties = 0

    done = False

    while not done:
        state_id = state[0]

        if random.uniform(0, 1) > epsilon:
            action = np.argmax(q_table[state_id])
        else:
            action = env.action_space.sample()

        print(f'Taken {action} action')

        new_state, reward, done, terminated, info = env.step(action)

        old_value = q_table[state_id][action]
        next_max = np.max(q_table[new_state])

        new_value = (
            (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        )
        q_table[state_id, action] = new_value

        if reward == -10:
            penalties += 1
