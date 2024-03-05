import numpy as np
from copy import deepcopy


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
