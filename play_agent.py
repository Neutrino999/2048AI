# play_agent.py

import numpy as np
from game_2048_env import Game2048Env
from dqn_agent import DQNAgent
import tensorflow as tf

if __name__ == "__main__":
    env = Game2048Env()
    state_shape = env.get_state().shape
    action_size = 4
    agent = DQNAgent(state_shape, action_size)
    # 加载训练好的模型（替换为你的模型文件名）
    agent.model = tf.keras.models.load_model(
        'models/model_100.h5',
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )
    agent.epsilon = 0  # 设置epsilon为0，完全利用

    state = env.reset()
    done = False

    previous_score = 0
    no_score_increase_steps = 0  # 连续未增加得分的步数
    max_no_increase_steps = 10   # 当连续10步得分未增加时停止

    while not done:
        env.render()  # 显示当前棋盘状态
        available_actions = env.get_available_actions()
        if not available_actions:
            print("没有可用的动作，游戏结束。")
            break
        action = agent.act(state, available_actions)
        next_state, reward, done, _ = env.step(action)
        state = next_state

        # 检查得分是否增加
        current_score = env.score
        if current_score > previous_score:
            no_score_increase_steps = 0  # 得分增加，重置计数器
        else:
            no_score_increase_steps += 1  # 得分未增加，计数器加一

        previous_score = current_score

        # 如果连续若干步得分未增加，停止程序
        if no_score_increase_steps >= max_no_increase_steps:
            print(f"得分在连续 {max_no_increase_steps} 步未增加，停止程序。")
            break

    env.render()
    print(f"Final Score: {env.score}")
