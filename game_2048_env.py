# game_2048_env.py

import numpy as np
import random

class Game2048Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.done = False
        self.add_new_tile()
        self.add_new_tile()
        return self.get_state()

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def get_state(self):
        # 将棋盘状态归一化处理（可选）
        state = np.log2(self.board + 1) / np.log2(2048)
        return state

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {}
        
        moved, score_gain = self.move(action)
        if moved:
            reward = score_gain  # 可以考虑增加额外的奖励，如空格数量
            self.add_new_tile()
            if not self.can_move():
                self.done = True
        else:
            reward = -5  # 对无效动作给予较大的惩罚
        self.score += score_gain
        return self.get_state(), reward, self.done, {}

    def get_available_actions(self):
        available_actions = []
        for action in range(4):
            temp_env = Game2048Env()
            temp_env.board = self.board.copy()
            moved, _ = temp_env.move(action)
            if moved:
                available_actions.append(action)
        return available_actions

    def render(self):
        print(f"Score: {self.score}")
        print('-' * 25)
        for row in self.board:
            print('|', end='')
            for num in row:
                if num == 0:
                    print('     |', end='')
                else:
                    print(f'{num:5d}|', end='')
            print('\n' + '-' * 25)

    def move(self, direction):
        original_board = self.board.copy()
        score_gain = 0
        moved = False

        # 根据方向定义移动函数
        if direction == 0:  # 上
            self.board = np.rot90(self.board, -1)
        elif direction == 1:  # 下
            self.board = np.rot90(self.board, 1)
        elif direction == 2:  # 左
            pass  # 不需要旋转
        elif direction == 3:  # 右
            self.board = np.rot90(self.board, 2)
        else:
            raise ValueError("Invalid action")

        for i in range(4):
            tight = self.board[i][self.board[i] != 0]
            merged = []
            skip = False
            j = 0
            while j < len(tight):
                if not skip and j + 1 < len(tight) and tight[j] == tight[j + 1]:
                    merged.append(tight[j] * 2)
                    score_gain += tight[j] * 2
                    skip = True
                else:
                    merged.append(tight[j])
                    skip = False
                j += 1 if not skip else 2
            merged.extend([0] * (4 - len(merged)))
            self.board[i] = merged
        # 将棋盘旋转回原始方向
        if direction == 0:
            self.board = np.rot90(self.board, 1)
        elif direction == 1:
            self.board = np.rot90(self.board, -1)
        elif direction == 3:
            self.board = np.rot90(self.board, 2)

        moved = not np.array_equal(self.board, original_board)
        return moved, score_gain

    def can_move(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    return True
                # 检查水平合并
                if j < 3 and self.board[i][j] == self.board[i][j + 1]:
                    return True
                # 检查垂直合并
                if i < 3 and self.board[i][j] == self.board[i + 1][j]:
                    return True
        return False

    def is_game_over(self):
        return not self.can_move()
