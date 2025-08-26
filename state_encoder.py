from collections import Counter # <-- ADD THIS LINE
import numpy as np
from scrabble_engine import ScrabbleGame, BOARD_SIZE, RACK_SIZE

# --- Constants for Encoding ---
# Map letters to an index (A=0, B=1, ..., Z=25, _=26)
LETTER_TO_INDEX = {chr(ord('A') + i): i for i in range(26)}
LETTER_TO_INDEX['_'] = 26
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}
NUM_LETTERS = 26

def encode_state(game: ScrabbleGame, move_to_evaluate=None):
    """
    Constructs the complete state tensor for LexiZeroNet.
    """
    # --- Part 1: Board Tensor ---
    board_tensor = np.zeros((85, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Handle the special case for the training loop where only a move is evaluated
    if not game:
        if move_to_evaluate:
            word, (start_r, start_c), orientation = move_to_evaluate
        
            r, c = start_r, start_c
            for letter in word:
                # Mark the square for this letter in the move
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    board_tensor[84, r, c] = 1.0
            
                # Move to the next square based on orientation
                if orientation == 'H':
                    c += 1
                else: # 'V'
                    r += 1
        # Return the board tensor and a dummy scalar vector
        return board_tensor, np.zeros(85, dtype=np.float32)

    # --- If we have a valid game object, proceed with full encoding ---
    player = game.get_current_player()
    opponent = game.get_opponent()

    # Channels 0-26: Letter and Blank Planes
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            tile = game.board.get_tile(r, c)
            if tile:
                board_tensor[LETTER_TO_INDEX[tile.letter], r, c] = 1.0
                if tile.is_blank:
                    board_tensor[LETTER_TO_INDEX['_'], r, c] = 1.0

    # Channels 27-30: Bonus Square Planes
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            board_tensor[27:31, r, c] = game.board.get_bonus_masks(r, c)
            
            
    # --- NEW: Channels 31-82: Cross-Set Planes ---
    dictionary_lookup = game.gaddag.dictionary
    for r_start in range(BOARD_SIZE):
        for c_start in range(BOARD_SIZE):
            if not game.board.is_empty(r_start, c_start):
                continue # Cross-sets are only for empty squares

            # Horizontal cross-sets (channels 31-56)
            # Find letters that can be placed at (r_start, c_start) to form a valid vertical word
            prefix = ""
            r = r_start - 1
            while r >= 0 and not game.board.is_empty(r, c_start):
                prefix = game.board.get_tile(r, c_start).letter + prefix
                r -= 1
            
            suffix = ""
            r = r_start + 1
            while r < BOARD_SIZE and not game.board.is_empty(r, c_start):
                suffix += game.board.get_tile(r, c_start).letter
                r += 1

            if prefix or suffix: # Only check if there's an adjacent tile
                for letter_idx in range(NUM_LETTERS):
                    letter = INDEX_TO_LETTER[letter_idx]
                    word = prefix + letter + suffix
                    if word in dictionary_lookup:
                        board_tensor[31 + letter_idx, r_start, c_start] = 1.0

            # Vertical cross-sets (channels 57-82)
            # Find letters that can be placed at (r_start, c_start) to form a valid horizontal word
            prefix = ""
            c = c_start - 1
            while c >= 0 and not game.board.is_empty(r_start, c):
                prefix = game.board.get_tile(r_start, c).letter + prefix
                c -= 1
            
            suffix = ""
            c = c_start + 1
            while c < BOARD_SIZE and not game.board.is_empty(r_start, c):
                suffix += game.board.get_tile(r_start, c).letter
                c += 1
            
            if prefix or suffix: # Only check if there's an adjacent tile
                for letter_idx in range(NUM_LETTERS):
                    letter = INDEX_TO_LETTER[letter_idx]
                    word = prefix + letter + suffix
                    if word in dictionary_lookup:
                        board_tensor[57 + letter_idx, r_start, c_start] = 1.0

    # Channel 83: Opponent's Last Move
    if game.last_move:
        for r, c, _ in game.last_move:
            board_tensor[83, r, c] = 1.0

    # Channel 84: Current Evaluated Move (if provided)
    if move_to_evaluate:
        # --- ADD THIS CHECK TO HANDLE DIFFERENT MOVE TYPES ---
        # Only try to unpack and draw scoring plays (3-element tuples)
        if isinstance(move_to_evaluate, tuple) and len(move_to_evaluate) == 3:
            word, (start_r, start_c), orientation = move_to_evaluate
            
            r, c = start_r, start_c
            for letter in word:
                # Mark the square for this letter in the move
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    board_tensor[84, r, c] = 1.0
                
                # Move to the next square based on orientation
                if orientation == 'H':
                    c += 1
                else: # 'V'
                    r += 1

    # --- Part 2: Scalar Vector ---
    scalar_features = []

    # Player Rack (27 scalars)
    player_rack_counts = np.zeros(NUM_LETTERS + 1, dtype=np.float32)
    for tile in player.rack:
        player_rack_counts[LETTER_TO_INDEX[tile.letter]] += 1
    scalar_features.extend(player_rack_counts / RACK_SIZE)

    # Belief State: Opponent's Rack and Bag (54 scalars)
    unseen_tile_counts = game.get_unseen_tile_counts()
    unseen_tiles_list = [letter for letter, count in unseen_tile_counts.items() for _ in range(count)]
    total_unseen = len(unseen_tiles_list)
    opponent_rack_size = len(opponent.rack)

    opp_rack_dist = np.zeros(NUM_LETTERS + 1, dtype=np.float32)
    bag_dist = np.zeros(NUM_LETTERS + 1, dtype=np.float32)

    if total_unseen > 0:
        unseen_prob = Counter(unseen_tiles_list)
        for letter, count in unseen_prob.items():
            prob = count / total_unseen
            opp_rack_dist[LETTER_TO_INDEX[letter]] = prob * opponent_rack_size
            bag_dist[LETTER_TO_INDEX[letter]] = prob

    scalar_features.extend(opp_rack_dist)
    scalar_features.extend(bag_dist)

    # Game Metadata (4 scalars)
    scalar_features.append(player.score / 500.0)
    scalar_features.append(opponent.score / 500.0)
    scalar_features.append(game.turn_number / 30.0)
    scalar_features.append(game.bag.get_remaining_count())

    scalar_vector = np.array(scalar_features, dtype=np.float32)

    # --- THIS IS THE CRUCIAL FIX ---
    return board_tensor, scalar_vector


# --- Example Usage ---
if __name__ == '__main__':
    # Create a game
    game = ScrabbleGame(player_names=["P1", "LexiZero"])
    
    # Simulate a first move for P1
    # Note: This is a fake move for demonstration, not generated by GADDAG
    move = [(7, 7, 'H'), (7, 8, 'I')]
    game.play_move(move) # P1 plays, turn passes to LexiZero

    # Now, encode the state from LexiZero's perspective
    print("Encoding state for LexiZero's turn...")
    board_t, scalar_v = encode_state(game)

    print(f"Board Tensor Shape: {board_t.shape}")
    print(f"Scalar Vector Shape: {scalar_v.shape}")

    # You can inspect parts of the tensor, e.g., the letter 'H' plane
    h_index = LETTER_TO_INDEX['H']
    print(f"\nValue at (7,7) on 'H' plane: {board_t[h_index, 7, 7]}")
    # Check opponent's last move plane
    print(f"Value at (7,7) on opponent's move plane: {board_t[83, 7, 7]}")

