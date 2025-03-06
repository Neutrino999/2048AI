# dqn_agent.py

import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # 状态形状，例如(4, 4)
        self.action_size = action_size  # 动作数量，2048游戏为4
        self.memory = deque(maxlen=2000)  # 经验回放缓冲区
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()

    def _build_model(self):
        # 构建神经网络模型，使用CNN处理棋盘
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(shape=self.state_shape))
        model.add(layers.Reshape((*self.state_shape, 1)))
        model.add(layers.Conv2D(128, (2, 2), activation='relu'))
        model.add(layers.Conv2D(128, (2, 2), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # 保存经验
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            # 随机选择有效的动作
            return random.choice(available_actions)
        q_values = self.model.predict(state[np.newaxis, ...])
        # 只选择有效动作中的最佳动作
        valid_q_values = [(action, q_values[0][action]) for action in available_actions]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]
        return best_action

    def replay(self):
        # 从记忆中随机抽取批量数据进行训练
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in minibatch])
        targets = self.model.predict(states)
        next_states = np.array([experience[3] for experience in minibatch])
        next_q_values = self.model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(next_q_values[i])

        self.model.fit(states, targets, epochs=1, verbose=0)
        # 逐步减少探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
