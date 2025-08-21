import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# Board convention (as in your code): 6 rows x 7 cols, values in {0=empty, 1=human, -1=AI}
# We'll search from the AI's perspective (AI pieces = -1). Negamax handles side to move.

WINDOWS = []  # list of lists of (r,c) positions for every length-4 line on the board
ROWS, COLS = 6, 7
for r in range(ROWS):
    for c in range(COLS-3):  # horizontal
        WINDOWS.append([(r,c+i) for i in range(4)])
for r in range(ROWS-3):
    for c in range(COLS):    # vertical
        WINDOWS.append([(r+i,c) for i in range(4)])
for r in range(ROWS-3):
    for c in range(COLS-3):  # diag /
        WINDOWS.append([(r+i,c+i) for i in range(4)])
for r in range(3, ROWS):
    for c in range(COLS-3):  # diag \
        WINDOWS.append([(r-i,c+i) for i in range(4)])

CENTER_COL = COLS // 2
MOVE_ORDER = [3, 4, 2, 5, 1, 6, 0]  # center-first ordering tends to prune more

@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int   # 0=exact, -1=alpha (lower bound), +1=beta (upper bound)
    best_move: Optional[int]

class BetterAIPlayer:
    """
    Very strong Connect-4 AI:
    - negamax + alpha-beta, iterative deepening
    - transposition table
    - heuristic scoring of all length-4 windows + center bias
    """
    def __init__(self, time_limit: float = 1.0, max_depth: int = 12):
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.tt: Dict[int, TTEntry] = {}
        self.start_time = 0.0
        self.stop = False
        # precompute powers for hashing
        self._zobrist = np.random.SeedSequence(12345).generate_state(ROWS*COLS*2, dtype=np.uint64).reshape(ROWS, COLS, 2)

    # --- Public API expected by your env ---
    def reset(self, **kwargs):
        pass

    def act(self, observation) -> int:
        """
        Called by the env to get the AI's move (column index 0..6).
        observation is a flattened 6x7 board (0 empty, 1 human, -1 AI).
        """
        board = np.array(observation, dtype=int).reshape(ROWS, COLS)
        # We assume it's AI's turn when act() is called by the env for opponent.
        return self.best_move(board, player=-1)

    # Some envs call "policy" or "get_action"; add alias to be safe.
    def policy(self, obs): return self.act(obs)
    def get_action(self, obs): return self.act(obs)

    # --- Search driver ---
    def best_move(self, board: np.ndarray, player: int) -> int:
        self.start_time = time.time()
        self.stop = False
        self.tt.clear()

        valid_moves = self._valid_moves(board)
        if not valid_moves:
            return 3  # fallback (shouldn't happen)

        best_col = valid_moves[0]
        best_score = -math.inf

        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if self._time_up(): break
            score, move = self._negamax_root(board, depth, player, -math.inf, math.inf)
            if self.stop: break
            if move is not None:
                best_col, best_score = move, score

        return best_col

    # --- Negamax with TT ---
    def _negamax_root(self, board: np.ndarray, depth: int, player: int, alpha: float, beta: float) -> Tuple[float, Optional[int]]:
        best_move = None
        best_score = -math.inf
        moves = self._ordered_moves(board)
        for col in moves:
            if self._time_up(): 
                self.stop = True
                break
            r = self._next_open_row(board, col)
            if r is None: 
                continue
            # Make move
            board[r, col] = player
            if self._is_win(board, player, r, col):
                score = 10_000_000  # immediate win
            else:
                score = -self._negamax(board, depth - 1, -player, -beta, -alpha)
            # Undo
            board[r, col] = 0

            if score > best_score:
                best_score = score
                best_move = col
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_move


    def play(self, board_input) -> int:
        """
        Gym env calls this with a 6x7 board (0 empty, 1=human, -1=AI).
        Return the chosen column (0..6).
        """
        # Accept list or numpy, flattened or 2D; never mutate caller's board.
        b = np.array(board_input, dtype=int)
        if b.size == 42:
            b = b.reshape(6, 7)
        elif b.shape != (6, 7):
            raise ValueError(f"Unexpected board shape: {b.shape} (expected 6x7)")
        return self.best_move(b.copy(), player=-1)


    def _negamax(self, board: np.ndarray, depth: int, player: int, alpha: float, beta: float) -> float:
        if self._time_up():
            self.stop = True
            return 0.0

        key = self._hash(board)
        if key in self.tt:
            entry = self.tt[key]
            if entry.depth >= depth:
                if entry.flag == 0:
                    return entry.score
                elif entry.flag < 0:
                    alpha = max(alpha, entry.score)
                else:
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

        valid = self._valid_moves(board)
        if not valid:
            return 0  # draw

        # terminal or depth limit
        if depth == 0:
            return self._evaluate(board) * (1 if player == -1 else -1)

        value = -math.inf
        best_flag = 1  # assume beta cut (upper bound) until proven exact
        best_move = None

        for col in self._order_with_tt_first(valid, key):
            r = self._next_open_row(board, col)
            if r is None: 
                continue
            board[r, col] = player
            if self._is_win(board, player, r, col):
                score = 10_000_000
            else:
                score = -self._negamax(board, depth - 1, -player, -beta, -alpha)
            board[r, col] = 0

            if score > value:
                value = score
                best_move = col
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        # store in TT
        flag = 0
        if value <= alpha:
            flag = 1    # upper bound (beta)
        elif value >= beta:
            flag = -1   # lower bound (alpha)
        self.tt[key] = TTEntry(depth=depth, score=value, flag=flag, best_move=best_move)
        return value

    # --- Helpers: moves, terminal checks, eval, hashing ---
    def _valid_moves(self, board: np.ndarray):
        return [c for c in range(COLS) if board[0, c] == 0]

    def _ordered_moves(self, board: np.ndarray):
        return [c for c in MOVE_ORDER if board[0, c] == 0]

    def _order_with_tt_first(self, valid, key):
        bm = self.tt.get(key).best_move if key in self.tt and self.tt[key].best_move in valid else None
        if bm is not None:
            return [bm] + [c for c in MOVE_ORDER if c != bm and c in valid]
        return [c for c in MOVE_ORDER if c in valid]

    def _next_open_row(self, board: np.ndarray, col: int) -> Optional[int]:
        col_vals = board[:, col]
        for r in range(ROWS - 1, -1, -1):
            if col_vals[r] == 0:
                return r
        return None

    def _is_win(self, board: np.ndarray, player: int, r: int, c: int) -> bool:
        # Check 4 directions that pass through (r,c)
        def count_dir(dr, dc):
            cnt = 0
            rr, cc = r, c
            while 0 <= rr < ROWS and 0 <= cc < COLS and board[rr, cc] == player:
                cnt += 1
                rr += dr; cc += dc
            return cnt
        # count (forward + backward - 1) in each axis
        for dr, dc in [(0,1), (1,0), (1,1), (-1,1)]:
            total = count_dir(dr, dc) + count_dir(-dr, -dc) - 1
            if total >= 4:
                return True
        return False

    # Heuristic: sum over all 4-length windows
    # Weights: 3-in-a-row with empty is huge, 2-in-a-row moderate, block opponent, center bias, etc.
    def _evaluate(self, board: np.ndarray) -> int:
        score = 0

        # Center control
        center_array = board[:, CENTER_COL]
        score += 6 * (np.count_nonzero(center_array == -1) - np.count_nonzero(center_array == 1))

        # Windows
        for window in WINDOWS:
            vals = [board[r, c] for (r, c) in window]
            ai = vals.count(-1)
            hu = vals.count(1)
            empty = vals.count(0)

            if ai > 0 and hu == 0:
                # Favor making 4
                if ai == 3 and empty == 1:
                    score += 1000
                elif ai == 2 and empty == 2:
                    score += 30
                elif ai == 1 and empty == 3:
                    score += 5
            elif hu > 0 and ai == 0:
                # Block human threats
                if hu == 3 and empty == 1:
                    score -= 1200  # slightly larger magnitude to prioritize blocks
                elif hu == 2 and empty == 2:
                    score -= 35
                elif hu == 1 and empty == 3:
                    score -= 5

        return score

    def _hash(self, board: np.ndarray) -> int:
        # Zobrist-like: two piece types (human=1 index 0, ai=-1 index 1)
        h = np.uint64(0)
        for r in range(ROWS):
            for c in range(COLS):
                v = board[r, c]
                if v == 1:
                    h ^= self._zobrist[r, c, 0]
                elif v == -1:
                    h ^= self._zobrist[r, c, 1]
        return int(h)

    def _time_up(self) -> bool:
        return (time.time() - self.start_time) >= self.time_limit
