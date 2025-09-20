#include <array>
#include <unordered_map>
#include <cstdint>
#include <limits>
#include <iostream>

using BoardMask = uint64_t;

// Constants
constexpr int WIDTH = 7;
constexpr int HEIGHT = 6;
constexpr int MIN_SCORE = -(WIDTH * HEIGHT) / 2 + 3;
constexpr int MAX_SCORE = (WIDTH * HEIGHT + 1) / 2 - 3;

// Bitboard representation
struct Position {
    BoardMask current_position = 0;  // player’s stones
    BoardMask mask = 0;              // all stones
    int moves = 0;                   // number of moves played
};

// Column bit indexing: each column has (HEIGHT+1) slots
constexpr int COL_HEIGHT = HEIGHT + 1;
constexpr BoardMask COL_MASK = (1ULL << COL_HEIGHT) - 1ULL;

// Precomputed top masks
std::array<BoardMask, WIDTH> TOP_MASKS = [] {
    std::array<BoardMask, WIDTH> arr{};
    for (int col = 0; col < WIDTH; ++col)
        arr[col] = 1ULL << (col * COL_HEIGHT + HEIGHT);
    return arr;
}();

// Precomputed bottom masks
std::array<BoardMask, WIDTH> BOTTOM_MASKS = [] {
    std::array<BoardMask, WIDTH> arr{};
    for (int col = 0; col < WIDTH; ++col)
        arr[col] = 1ULL << (col * COL_HEIGHT);
    return arr;
}();

// Helpers
inline bool can_play(const Position &pos, int col) {
    return (pos.mask & TOP_MASKS[col]) == 0;
}

inline void play(Position &pos, int col) {
    pos.current_position ^= pos.mask;
    pos.mask |= pos.mask + BOTTOM_MASKS[col];
    pos.moves++;
}

inline bool is_winning_move(const Position &pos, int col) {
    BoardMask pos2 = pos.current_position;
    pos2 |= (pos.mask + BOTTOM_MASKS[col]) & (COL_MASK << (col * COL_HEIGHT));
    auto test = [&](int shift) {
        BoardMask m = pos2 & (pos2 >> shift);
        return (m & (m >> (2 * shift))) != 0;
    };
    return test(1) || test(COL_HEIGHT) || test(COL_HEIGHT - 1) || test(COL_HEIGHT + 1);
}

// Transposition table
struct TTEntry {
    int16_t val;
    int8_t depth;
};
std::unordered_map<BoardMask, TTEntry> trans_table;

// Node counter
static size_t nodes = 0;

// Negamax with alpha-beta + depth cutoff
int negamax(Position pos, int alpha, int beta, int depth = 0) {
    ++nodes;

    if (nodes % 1000000 == 0 && depth == 0) {
        std::cout << "[INFO] " << nodes << " nodes searched...\n";
    }

    // Depth cutoff for debugging (remove later for full perfect play)
    // if (depth > 10) return 0;

    if (pos.moves == WIDTH * HEIGHT) return 0;

    auto key = pos.current_position + pos.mask;
    if (auto it = trans_table.find(key); it != trans_table.end()) {
        const TTEntry &entry = it->second;
        if (entry.depth >= WIDTH * HEIGHT - pos.moves)
            return entry.val;
    }

    for (int col = 0; col < WIDTH; ++col)
        if (can_play(pos, col) && is_winning_move(pos, col))
            return (WIDTH * HEIGHT + 1 - pos.moves) / 2;

    int max_score = (WIDTH * HEIGHT - 1 - pos.moves) / 2;
    if (beta > max_score) {
        beta = max_score;
        if (alpha >= beta) return beta;
    }

    int best = MIN_SCORE;
    for (int col = 3, d = 0; d < WIDTH; d++, col = (3 + ((d & 1) ? -((d + 1) / 2) : (d / 2)))) {
        if (can_play(pos, col)) {
            Position new_pos = pos;
            play(new_pos, col);

            int score = -negamax(new_pos, -beta, -alpha, depth + 1);

            // if (depth < 2) { // only show top levels
            //     std::cout << std::string(depth * 2, ' ')
            //               << "[Depth " << depth << "] Col " << col
            //               << " → Score " << score
            //               << " (alpha=" << alpha << ", beta=" << beta << ")\n";
            // }

            if (score >= beta) {
                trans_table[key] = { (int16_t)score, (int8_t)(WIDTH * HEIGHT - pos.moves) };
                return score;
            }
            if (score > best) best = score;
            if (score > alpha) alpha = score;
        }
    }

    trans_table[key] = { (int16_t)best, (int8_t)(WIDTH * HEIGHT - pos.moves) };
    return best;
}

// Get best move
int best_move(Position pos) {
    int best_col = -1;
    int best_val = MIN_SCORE;
    nodes = 0; // reset counter
    if (pos.moves == 0) return 3;

    for (int col = 3, d = 0; d < WIDTH; d++, col = (3 + ((d & 1) ? -((d + 1) / 2) : (d / 2)))) {
        if (can_play(pos, col)) {
            Position new_pos = pos;
            play(new_pos, col);
            if (is_winning_move(pos, col)) return col; // immediate win
            int val = -negamax(new_pos, -MAX_SCORE, -best_val, 1);
            if (val > best_val) {
                best_val = val;
                best_col = col;
            }
        }
    }
    std::cout << "[INFO] Total nodes searched: " << nodes << "\n";
    return best_col;
}

// ----------------- Demo utilities -----------------
void print_board(const Position &pos) {
    char grid[HEIGHT][WIDTH];
    for (int c = 0; c < WIDTH; ++c) {
        BoardMask colMask = (COL_MASK << (c * COL_HEIGHT));
        BoardMask stones = pos.mask & colMask;
        for (int r = 0; r < HEIGHT; ++r) {
            BoardMask bit = 1ULL << (c * COL_HEIGHT + r);
            if (!(stones & bit)) grid[HEIGHT-1-r][c] = '.';
            else if (pos.current_position & bit) grid[HEIGHT-1-r][c] = 'X';
            else grid[HEIGHT-1-r][c] = 'O';
        }
    }
    std::cout << "\n  0 1 2 3 4 5 6\n";
    for (int r = 0; r < HEIGHT; ++r) {
        std::cout << r << " ";
        for (int c = 0; c < WIDTH; ++c)
            std::cout << grid[r][c] << " ";
        std::cout << "\n";
    }
}

// Demo main
int main() {
    // Reserve TT memory for speed
    trans_table.reserve(1 << 22);
    trans_table.max_load_factor(0.7f);

    Position pos;
    while (true) {
        int col = best_move(pos);
        std::cout << "\nAI plays: " << col << "\n";
        play(pos, col);
        print_board(pos);
        if (is_winning_move(pos, col)) {
            std::cout << "AI wins!\n"; break;
        }
        int human;
        std::cout << "Your move (0-6): ";
        std::cin >> human;
        if (!can_play(pos, human)) {
            std::cout << "Column full/illegal, try again.\n";
            continue;
        }
        play(pos, human);
        print_board(pos);
        if (is_winning_move(pos, human)) {
            std::cout << "You win!\n"; break;
        }
    }
}
