import random
from collections import Counter
import copy

# --- Constants ---

# Standard Scrabble tile distribution and values
TILE_DISTRIBUTION = {
    'A': (9, 1), 'B': (2, 3), 'C': (2, 3), 'D': (4, 2), 'E': (12, 1),
    'F': (2, 4), 'G': (3, 2), 'H': (2, 4), 'I': (9, 1), 'J': (1, 8),
    'K': (1, 5), 'L': (4, 1), 'M': (2, 3), 'N': (6, 1), 'O': (8, 1),
    'P': (2, 3), 'Q': (1, 10), 'R': (6, 1), 'S': (4, 1), 'T': (6, 1),
    'U': (4, 1), 'V': (2, 4), 'W': (2, 4), 'X': (1, 8), 'Y': (2, 4),
    'Z': (1, 10), '_': (2, 0)  # Represents a blank tile
}

BOARD_SIZE = 15
RACK_SIZE = 7

# Bonus square layout (Triple Word, Double Word, Triple Letter, Double Letter)
BONUS_SQUARES = {
    (0, 0): "TW", (0, 7): "TW", (0, 14): "TW",
    (7, 0): "TW", (7, 14): "TW",
    (14, 0): "TW", (14, 7): "TW", (14, 14): "TW",

    (1, 1): "DW", (2, 2): "DW", (3, 3): "DW", (4, 4): "DW",
    (1, 13): "DW", (2, 12): "DW", (3, 11): "DW", (4, 10): "DW",
    (10, 4): "DW", (11, 3): "DW", (12, 2): "DW", (13, 1): "DW",
    (10, 10): "DW", (11, 11): "DW", (12, 12): "DW", (13, 13): "DW",
    (7, 7): "DW", # Center square is a Double Word

    (1, 5): "TL", (1, 9): "TL",
    (5, 1): "TL", (5, 5): "TL", (5, 9): "TL", (5, 13): "TL",
    (9, 1): "TL", (9, 5): "TL", (9, 9): "TL", (9, 13): "TL",
    (13, 5): "TL", (13, 9): "TL",

    (0, 3): "DL", (0, 11): "DL",
    (2, 6): "DL", (2, 8): "DL",
    (3, 0): "DL", (3, 7): "DL", (3, 14): "DL",
    (6, 2): "DL", (6, 6): "DL", (6, 8): "DL", (6, 12): "DL",
    (7, 3): "DL", (7, 11): "DL",
    (8, 2): "DL", (8, 6): "DL", (8, 8): "DL", (8, 12): "DL",
    (11, 0): "DL", (11, 7): "DL", (11, 14): "DL",
    (12, 6): "DL", (12, 8): "DL",
    (14, 3): "DL", (14, 11): "DL"
}

# --- Core Classes ---

class Tile:
    """Represents a single Scrabble tile."""
    def __init__(self, letter, value):
        self.letter = letter
        self.value = value
        self.is_blank = (letter == '_')

    def __repr__(self):
        return f"{self.letter}({self.value})"

class Bag:
    """Manages the collection of all tiles for a game."""
    def __init__(self):
        """Initializes and shuffles the bag of tiles."""
        self.tiles = self._create_tiles()
        self.shuffle()

    def _create_tiles(self):
        """Creates the full set of 100 Scrabble tiles."""
        tiles = []
        for letter, (count, value) in TILE_DISTRIBUTION.items():
            for _ in range(count):
                tiles.append(Tile(letter, value))
        return tiles

    def shuffle(self):
        """Randomly shuffles the tiles in the bag."""
        random.shuffle(self.tiles)

    def draw_tiles(self, num_tiles):
        """Draws a specified number of tiles from the bag."""
        drawn_tiles = []
        for _ in range(num_tiles):
            if not self.is_empty():
                drawn_tiles.append(self.tiles.pop())
            else:
                break
        return drawn_tiles

    def is_empty(self):
        """Checks if the bag is empty."""
        return len(self.tiles) == 0

    def get_remaining_count(self):
        """Returns the number of tiles left in the bag."""
        return len(self.tiles)
        
    def return_tiles(self, tiles):
        """Returns tiles to the bag and shuffles."""
        self.tiles.extend(tiles)
        self.shuffle()

class Board:
    """Represents the 15x15 Scrabble board."""
    def __init__(self):
        """Initializes an empty board."""
        self.grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.bonus_squares = BONUS_SQUARES
        self.size = BOARD_SIZE

    def place_tile(self, tile, row, col):
        """Places a tile on the board at a given position."""
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            self.grid[row][col] = tile
        else:
            raise ValueError("Invalid board position.")

    def get_tile(self, row, col):
        """Retrieves the tile at a given position."""
        return self.grid[row][col]

    def is_empty(self, row, col):
        """Checks if a given square is empty."""
        return self.grid[row][col] is None
        
    def get_bonus_masks(self, y, x):
        """Returns a 4-element list representing bonus squares for the encoder."""
        bonus = self.bonus_squares.get((y, x))
        return [
            1.0 if bonus == "DL" else 0.0,
            1.0 if bonus == "TL" else 0.0,
            1.0 if bonus == "DW" else 0.0,
            1.0 if bonus == "TW" else 0.0,
        ]

    def display(self):
        """Prints a simple text-based representation of the board."""
        for row in range(BOARD_SIZE):
            row_str = ""
            for col in range(BOARD_SIZE):
                tile = self.grid[row][col]
                if tile:
                    row_str += f" {tile.letter} "
                else:
                    bonus = self.bonus_squares.get((row, col))
                    if bonus:
                        row_str += f"{bonus:^3}"
                    else:
                        row_str += " . "
            print(row_str)

class Player:
    """Represents a player in the game."""
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.rack = []

    def add_tiles_to_rack(self, tiles):
        """Adds a list of tiles to the player's rack."""
        self.rack.extend(tiles)

    def remove_tiles_from_rack(self, letters_to_remove):
        """Removes tiles corresponding to the given letters from the rack."""
        temp_rack = self.rack[:]
        removed_tiles = []
        for letter in letters_to_remove:
            found = False
            for tile in temp_rack:
                if tile.letter == letter:
                    removed_tiles.append(tile)
                    temp_rack.remove(tile)
                    found = True
                    break
            if not found:
                # Handle blank tiles if a specific letter is not found
                for tile in temp_rack:
                    if tile.letter == '_':
                        blank_tile = Tile(letter, 0)
                        removed_tiles.append(blank_tile)
                        temp_rack.remove(tile)
                        found = True
                        break
            if not found:
                # If we still can't find the letter, this is an invalid move
                raise ValueError(f"Letter '{letter}' not available in rack: {[t.letter for t in self.rack]}")
        self.rack = temp_rack
        return removed_tiles

    def __repr__(self):
        return f"Player({self.name}, Score: {self.score}, Rack: {self.rack})"
        
    def exchange_tiles(self, letters_to_exchange, bag):
        """Exchanges specified tiles with the bag."""
        if bag.get_remaining_count() < len(letters_to_exchange):
            raise ValueError("Not enough tiles in the bag to exchange.")
        
        # Remove tiles from rack
        removed_tiles = self.remove_tiles_from_rack(letters_to_exchange)
        
        # Draw new tiles
        new_tiles = bag.draw_tiles(len(letters_to_exchange))
        self.add_tiles_to_rack(new_tiles)
        
        # Return old tiles to bag
        bag.return_tiles(removed_tiles)

class ScrabbleGame:
    def __init__(self, player_names, gaddag):
        if len(player_names) != 2:
            raise ValueError("Scrabble is a 2-player game.")
        self.board = Board()
        self.bag = Bag()
        self.players = [Player(name) for name in player_names]
        self.gaddag = gaddag
        self.current_player_index = 0
        self.turn_number = 1
        self.last_move = None
        self.consecutive_passes = 0
        self._initialize_game()

    def _initialize_game(self):
        """Gives initial racks to all players."""
        for player in self.players:
            player.add_tiles_to_rack(self.bag.draw_tiles(RACK_SIZE))

    def pass_turn(self):
        """Handles a player passing their turn."""
        self.consecutive_passes += 1
        self.next_turn()

    def play_move(self, move):
        """
        Plays a move - handles both formats:
        1. GADDAG format: (word, (start_r, start_c), orientation)  
        2. Legacy format: [(r, c, letter), ...]
        """
        player = self.get_current_player()
        
        # --- ADD THIS ELIF BLOCK ---
        # Handle Exchange format: ('EXCHANGE', 'LETTERS')
        if isinstance(move, tuple) and move[0] == 'EXCHANGE':
            letters_to_exchange = list(move[1])
            player.exchange_tiles(letters_to_exchange, self.bag)
            self.last_move = None # An exchange is not a scoring move
            self.consecutive_passes += 1 # Exchanging is like passing in terms of scoring
            self.next_turn()
            return 0 # An exchange scores 0 points
        
        # Handle GADDAG format: (word, (start_r, start_c), orientation)
        if isinstance(move, tuple) and len(move) == 3:
            word, (start_r, start_c), orientation = move
            
            # Convert GADDAG format to legacy format
            move_for_scoring = []
            letters_to_remove = []
            r, c = start_r, start_c
            
            for letter in word:
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    if self.board.is_empty(r, c):
                        letters_to_remove.append(letter)
                    move_for_scoring.append((r, c, letter))
                    
                    # Move to next position
                    if orientation == 'H':
                        c += 1
                    else:  # 'V'
                        r += 1
                else:
                    raise ValueError(f"Move extends outside board boundaries")
            
            
            # Remove tiles from rack
            played_tiles = player.remove_tiles_from_rack(letters_to_remove)
            
            # Place tiles on board
            r, c = start_r, start_c
            tile_index = 0
            for letter in word:
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    if self.board.is_empty(r, c):
                        tile_to_place = played_tiles[tile_index]
                        if tile_to_place.letter == '_':
                            tile_to_place.letter = letter
                        self.board.place_tile(tile_to_place, r, c)
                        tile_index += 1
                    
                    if orientation == 'H':
                        c += 1
                    else:
                        r += 1
        
        # Handle legacy format: [(r, c, letter), ...]
        elif isinstance(move, list):
            move_for_scoring = move
            letters_to_remove = []
            for r, c, letter in move:
                if self.board.is_empty(r, c):
                    letters_to_remove.append(letter)
            
            played_tiles = player.remove_tiles_from_rack(letters_to_remove)
            
            tile_index = 0
            for r, c, letter in move:
                if self.board.is_empty(r, c):
                    tile_to_place = played_tiles[tile_index]
                    if tile_to_place.letter == '_':
                        tile_to_place.letter = letter
                    self.board.place_tile(tile_to_place, r, c)
                    tile_index += 1
        else:
            raise ValueError(f"Invalid move format: {move}")
        
        # Update game state
        self.last_move = move_for_scoring
        move_score = self._calculate_score(move_for_scoring)
        if move_score > 0:
            self.consecutive_passes = 0
        player.score += move_score
        
        # Refill rack
        num_to_draw = RACK_SIZE - len(player.rack)
        player.add_tiles_to_rack(self.bag.draw_tiles(num_to_draw))
        self.next_turn()
        return move_score

    def is_game_over(self):
        """Checks for game-ending conditions."""
        # Standard end condition
        if self.bag.is_empty():
            for player in self.players:
                if not player.rack:
                    return True
        
        # End the game after 6 consecutive passes
        if self.consecutive_passes >= 6:
            return True
            
        return False

    def _calculate_score(self, move):
        """Calculates the total score for a move by finding all words formed."""
        if not move:
            return 0

        placed_coords = {(r, c) for r, c, _ in move}
        scored_words = set()
        total_score = 0
        
        # Find all unique words that contain at least one newly placed tile
        for r_placed, c_placed, _ in move:
            
            # Find horizontal word containing this placed tile
            start_col = c_placed
            while start_col > 0 and self.board.get_tile(r_placed, start_col - 1) is not None:
                start_col -= 1
            
            end_col = c_placed
            while end_col < BOARD_SIZE - 1 and self.board.get_tile(r_placed, end_col + 1) is not None:
                end_col += 1
            
            # Only score if it's a multi-letter word
            if end_col > start_col:
                word_key = ('H', r_placed, start_col, end_col)
                if word_key not in scored_words:
                    word_coords = [(r_placed, c) for c in range(start_col, end_col + 1)]
                    word_score = self._score_word(word_coords, placed_coords)
                    total_score += word_score
                    scored_words.add(word_key)
            
            # Find vertical word containing this placed tile
            start_row = r_placed
            while start_row > 0 and self.board.get_tile(start_row - 1, c_placed) is not None:
                start_row -= 1
            
            end_row = r_placed
            while end_row < BOARD_SIZE - 1 and self.board.get_tile(end_row + 1, c_placed) is not None:
                end_row += 1
            
            # Only score if it's a multi-letter word
            if end_row > start_row:
                word_key = ('V', c_placed, start_row, end_row)
                if word_key not in scored_words:
                    word_coords = [(r, c_placed) for r in range(start_row, end_row + 1)]
                    word_score = self._score_word(word_coords, placed_coords)
                    total_score += word_score
                    scored_words.add(word_key)
        
        # Add bingo bonus if all 7 tiles were used
        if len(placed_coords) == 7:
            total_score += 50
        
        return total_score
        
    def get_current_player(self):
        """Returns the player whose turn it is."""
        return self.players[self.current_player_index]

    def next_turn(self):
        """Advances the game to the next player's turn."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.turn_number += 1
        
    def get_opponent(self):
        """Returns the opponent player object."""
        return self.players[(self.current_player_index + 1) % len(self.players)]
        
    def get_unseen_tile_counts(self):
        """Calculates the counts of all tiles not visible to the current player."""
        unseen_counts = {letter: data[0] for letter, data in TILE_DISTRIBUTION.items()}

        # Subtract tiles on the current player's rack
        player = self.get_current_player()
        for tile in player.rack:
            original_letter = '_' if tile.is_blank else tile.letter
            if unseen_counts[original_letter] > 0:
                unseen_counts[original_letter] -= 1

        # Subtract tiles on the board
        for r in range(self.board.size):
            for c in range(self.board.size):
                tile = self.board.get_tile(r, c)
                if tile:
                    original_letter = '_' if tile.is_blank else tile.letter
                    if unseen_counts[original_letter] > 0:
                        unseen_counts[original_letter] -= 1
        
        return unseen_counts

    def _handle_end_game_scoring(self):
        """Calculates final scores based on unplayed tiles."""
        player_out = None
        for p in self.players:
            if not p.rack:
                player_out = p
                break
        
        if player_out:
            opponent = self.players[(self.players.index(player_out) + 1) % 2]
            rack_sum = sum(tile.value for tile in opponent.rack)
            player_out.score += rack_sum
            opponent.score -= rack_sum
        else:
            for p in self.players:
                p.score -= sum(tile.value for tile in p.rack)

    def get_outcome(self):
        """Determines the game outcome from the current player's perspective."""
        p1_temp = copy.deepcopy(self.players[0])
        p2_temp = copy.deepcopy(self.players[1])
        
        temp_game = copy.deepcopy(self)
        temp_game.players = [p1_temp, p2_temp]
        temp_game._handle_end_game_scoring()

        current_player_final_score = temp_game.players[self.current_player_index].score
        opponent_final_score = temp_game.players[(self.current_player_index + 1) % 2].score

        if current_player_final_score > opponent_final_score:
            return 1.0
        elif current_player_final_score < opponent_final_score:
            return -1.0
        else:
            return 0.0
            
    def _score_word(self, word_coords, placed_coords):
        """Scores a single word based on its coordinates."""
        letter_score = 0
        word_multiplier = 1
        
        for r, c in word_coords:
            tile = self.board.get_tile(r, c)
            if tile is None:
                continue
                
            tile_value = tile.value
            
            # Apply bonuses only to newly placed tiles
            if (r, c) in placed_coords:
                bonus = self.board.bonus_squares.get((r, c))
                if bonus == "DL":
                    tile_value *= 2
                elif bonus == "TL":
                    tile_value *= 3
                elif bonus == "DW":
                    word_multiplier *= 2
                elif bonus == "TW":
                    word_multiplier *= 3
            
            letter_score += tile_value
        
        return letter_score * word_multiplier
        
    def clone(self):
        """Creates a deep copy of the game state."""
        return copy.deepcopy(self)


# --- Example Usage ---
if __name__ == '__main__':
    print("Initializing Scrabble Game Engine...")
    # Create a test dictionary
    dict_file = "test_dict.txt"
    with open(dict_file, "w") as f:
        f.write("HELLO\nWORLD\nTEST\n")
    
    # Import Gaddag for the example
    from gaddag import Gaddag
    gaddag_for_example = Gaddag(dict_file)
    game = ScrabbleGame(player_names=["Player 1", "LexiZero"], gaddag=gaddag_for_example)
    
    print(f"Turn {game.turn_number}, Bag has {game.bag.get_remaining_count()} tiles left.")
    
    p1 = game.get_current_player()
    print(f"It's {p1.name}'s turn.")
    print(f"Rack: {p1.rack}")
    
    # Clean up test file
    import os
    os.remove(dict_file)