# train_agent.py

import numpy as np
from game_2048_env import Game2048Env
from dqn_agent import DQNAgent

if __name__ == "__main__":
    env = Game2048Env()
    state_shape = env.get_state().shape  # (4, 4)
    action_size = 4  # 上、下、左、右
    agent = DQNAgent(state_shape, action_size)
    episodes = 100  # 训练的轮数

    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            action = agent.act(state, available_actions)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

        print(f"Episode {e+1}/{episodes}, Score: {env.score}, Epsilon: {agent.epsilon:.2f}")

        # 每隔100轮保存一次模型
        if (e + 1) % 10 == 0:
            agent.model.save(f"models/model_{e+1}.h5")
