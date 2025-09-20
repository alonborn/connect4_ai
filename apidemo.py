#!/usr/bin/env python3
import requests

API = "https://kevinalbs.com/connect4/back-end/index.php/getMoves"

# Board: 6 rows x 7 cols, bottom row is r=0 (discs "fall" up)
EMPTY, P1, P2 = 0, 1, 2   # AI = P1 (X), Human = P2 (O)

def encode_board(board):
    # API wants top->bottom, left->right, '0','1','2'
    chars = []
    for r in range(5, -1, -1):       # top to bottom
        for c in range(7):           # left to right
            chars.append(str(board[r][c]))
    return "".join(chars)

def print_board(board):
    # ANSI color codes
    RED    = "\033[31m"
    BLUE   = "\033[34m"
    YELLOW = "\033[33m"
    RESET  = "\033[0m"

    # Column headers in yellow
    print("\n  " + " ".join(f"{YELLOW}{c}{RESET}" for c in range(7)))
    for r in range(5, -1, -1):  # top row down to bottom
        row = []
        for c in range(7):
            v = board[r][c]
            if v == EMPTY:
                row.append(".")
            elif v == P1:
                row.append(f"{RED}X{RESET}")
            else:
                row.append(f"{BLUE}O{RESET}")
        print(f"{r} " + " ".join(row))


def can_play(board, col):
    return 0 <= col <= 6 and board[5][col] == EMPTY

def play(board, col, player):
    for r in range(6):
        if board[r][col] == EMPTY:
            board[r][col] = player
            return True
    return False

def has_won(board, player):
    # horizontal
    for r in range(6):
        for c in range(7 - 3):
            if all(board[r][c + k] == player for k in range(4)):
                return True
    # vertical
    for c in range(7):
        for r in range(6 - 3):
            if all(board[r + k][c] == player for k in range(4)):
                return True
    # diag up-right
    for r in range(6 - 3):
        for c in range(7 - 3):
            if all(board[r + k][c + k] == player for k in range(4)):
                return True
    # diag up-left
    for r in range(6 - 3):
        for c in range(3, 7):
            if all(board[r + k][c - k] == player for k in range(4)):
                return True
    return False

def is_draw(board):
    return all(board[5][c] != EMPTY for c in range(7))

def best_move_from_api(board, player=P1):
    bstr = encode_board(board)
    resp = requests.get(API, params={"board_data": bstr, "player": player}, timeout=10)
    resp.raise_for_status()
    scores = resp.json()  # {"0":score0,...,"6":score6}
    playable = [c for c in range(7) if can_play(board, c)]
    if not playable:
        return -1, scores
    # pick best legal column; use get(..., -inf) in case a key is missing
    best = max(playable, key=lambda c: float(scores.get(str(c), float("-inf"))))
    return best, scores

def main():
    board = [[EMPTY for _ in range(7)] for _ in range(6)]
    print("Connect-4 API demo. AI = X (goes first). You = O.")

    while True:
        # --- AI turn ---
        ai_col, scores = best_move_from_api(board, player=P1)
        if ai_col == -1:
            print("No legal AI moves. Draw.")
            break
        play(board, ai_col, P1)
        print(f"\nAI plays: {ai_col}  (scores={scores})")
        print_board(board)
        if has_won(board, P1):
            print("AI wins! 🎉")
            break
        if is_draw(board):
            print("Draw.")
            break

        # --- Human turn ---
        while True:
            try:
                col = int(input("Your move (0-6): ").strip())
            except ValueError:
                print("Enter a number 0–6."); continue
            if can_play(board, col):
                break
            print("Illegal column, try again.")
        play(board, col, P2)
        print_board(board)
        if has_won(board, P2):
            print("You win! 👏")
            break
        if is_draw(board):
            print("Draw.")
            break

if __name__ == "__main__":
    main()
