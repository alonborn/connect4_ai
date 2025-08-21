import numpy as np
from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import AdultSmarterPlayer

def render_board_sideways(board):
    """
    Pretty-print the board in a vertical Connect Four style.
    Board is 6x7 with values 0 (empty), 1 (human), -1 (AI).
    """
    symbols = {0: ".", 1: "X", -1: "O"}  # FIX: handle -1 for AI
    rows, cols = board.shape

    print("\n   0   1   2   3   4   5   6")
    print("  ----------------------------")
    for r in range(rows):
        row_str = " | ".join(symbols[int(val)] for val in board[r])
        print(f"{r} | {row_str} |")
    print("  ----------------------------")

def get_board_from_obs(obs):
    """Convert observation (flattened array) to 6x7 board"""
    return np.array(obs).reshape(6, 7)

def main():
    env = ConnectFourEnv(opponent=AdultSmarterPlayer())
    obs, _ = env.reset()
    done = False

    print("🎮 Connect4 Game Started! You are Player 1 (X). The AI is Player 2 (O).")
    env.render()

    while not done:
        # Human move
        board = get_board_from_obs(obs)
        render_board_sideways(board)

        valid_move = False
        while not valid_move:
            try:
                col = int(input("Your move (0-6): "))
                if col < 0 or col > 6:
                    print("❌ Invalid column. Must be 0-6.")
                    continue
                if board[0, col] != 0:
                    print("❌ Column is full.")
                    continue
                valid_move = True
            except ValueError:
                print("❌ Please enter a number (0-6).")

        # This step includes both: your move + AI's response
        obs, reward, terminated, truncated, _ = env.step(col)

        board = get_board_from_obs(obs)
        render_board_sideways(board)

        if terminated or truncated:
            done = True


    # Final result
    if reward == 1:
        print("🎉 You win!")
    elif reward == -1:
        print("💀 AI wins!")
    else:
        print("🤝 It's a draw!")

if __name__ == "__main__":
    main()
