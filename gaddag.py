import os
from collections import defaultdict, Counter 

# --- Mock/Placeholder Classes for Standalone Testing ---
# In a real project, these would be imported from scrabble_engine.py
class Tile:
    def __init__(self, letter, value):
        self.letter = letter
        self.value = value
    def __repr__(self):
        return f"{self.letter}"

class Board:
    """A mock Board class for standalone testing of the GADDAG."""
    def __init__(self):
        self.size = 15
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
    def is_empty(self, r, c):
        return self.grid[r][c] is None
    def get_tile(self, r, c):
        return self.grid[r][c]
    def place_tile(self, tile, r, c):
        self.grid[r][c] = tile

# --- GADDAG Implementation ---
class GaddagNode:
    """A node in the GADDAG data structure."""
    def __init__(self):
        self.children = defaultdict(GaddagNode)
        self.is_terminal = False

class Gaddag:
    """
    GADDAG data structure for efficient and valid Scrabble move generation.
    This implementation includes cross-word validation and maintains compatibility
    with the original POMCTS interface.
    """
    BREAK_CHAR = '+'

    def __init__(self, dictionary_path):
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"Dictionary file not found at: {dictionary_path}")
        self.root = GaddagNode()
        self.dictionary = set() # For fast cross-word lookups
        self.dictionary_path = dictionary_path
        self._build(dictionary_path)

    def _build(self, dictionary_path):
        """Reads the dictionary, builds the GADDAG, and populates the lookup set."""
        with open(dictionary_path, 'r') as f:
            for word in f:
                word = word.strip().upper()
                if word:
                    self.dictionary.add(word)
                    self._add_word(word)

    def _add_word(self, word):
        """Adds a word and all its GADDAG permutations to the trie."""
        for i in range(len(word)):
            prefix = word[:i][::-1]
            suffix = word[i:]
            path = prefix + self.BREAK_CHAR + suffix
            node = self.root
            for char in path:
                node = node.children[char]
            node.is_terminal = True

    def find_moves(self, rack, board):
        """Finds all possible moves for a given rack and board state."""
        moves = set()
        anchors = self._find_anchors(board)
        rack_str = "".join([tile.letter for tile in rack])
        
        
        if not rack_str:
            return []

        for r, c in anchors:
            initial_moves_count = len(moves)
            
            # Generate moves extending from the anchor
            self._gen(rack_str, self.root, [], board, r, c, True, moves, rack_str)  # Horizontal
            self._gen(rack_str, self.root, [], board, r, c, False, moves, rack_str) # Vertical
            
            new_moves = len(moves) - initial_moves_count
            
        
        
        # Final validation: ensure all returned moves can actually be made with the given rack
        valid_moves = []
        invalid_count = 0
        for move in moves:
            is_valid = self._can_make_move_with_rack(move, rack_str, board) # Check if valid

            # --- ADD THIS IF-STATEMENT FOR DEBUGGING ---
            if not is_valid:
                print(f"GADDAG FAILED VALIDATION: Move {move} for rack '{rack_str}'")
            # --- END OF ADDED CODE ---

            if is_valid:
                valid_moves.append(move)
            else:
                invalid_count += 1

            valid_moves = []
        for move in moves:
            if self._can_make_move_with_rack(move, rack_str, board):
                valid_moves.append(move)

        return valid_moves
        
    def _can_make_move_with_rack(self, move, rack_str, board):
        """
        Validates that a move can actually be made with the given rack.
        This is a corrected, more robust version.
        """
        try:
            word, (start_r, start_c), orientation = move
            rack_counts = Counter(rack_str)
            blanks_available = rack_counts.pop('_', 0) # Get blanks and remove them

            # Determine which letters are needed from the rack
            letters_needed = []
            r, c = start_r, start_c
            for letter in word:
                if not (0 <= r < board.size and 0 <= c < board.size):
                    return False # Move goes off the board

                if board.is_empty(r, c):
                    letters_needed.append(letter)

                # Move to the next square
                r, c = (r, c + 1) if orientation == 'H' else (r + 1, c)

            # Check if the rack (without blanks) can satisfy the needed letters
            needed_counts = Counter(letters_needed)

            blanks_used = 0
            for letter, count_needed in needed_counts.items():
                count_on_rack = rack_counts.get(letter, 0)
                if count_on_rack < count_needed:
                    blanks_used += count_needed - count_on_rack

            return blanks_used <= blanks_available

        except Exception:
            # If any error occurs during validation, treat the move as invalid.
            return False
            
            # Check if we have enough letters (including blanks)
            needed_counts = Counter(letters_needed)
            blanks_available = rack_counts.get('_', 0)
            blanks_needed = 0
            
            for letter, count in needed_counts.items():
                available = rack_counts.get(letter, 0)
                if available < count:
                    blanks_needed += count - available
            
            return blanks_needed <= blanks_available
            
        except Exception:
            return False    

    def _find_anchors(self, board):
        """Identifies all anchor squares on the board."""
        anchors = set()
        if board.is_empty(7, 7):
            return [(7, 7)]
        for r in range(board.size):
            for c in range(board.size):
                if not board.is_empty(r, c):
                    continue # An anchor must be an empty square
                
                is_adjacent_to_tile = False
                # Check adjacent squares
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.size and 0 <= nc < board.size and not board.is_empty(nr, nc):
                        is_adjacent_to_tile = True
                        break
                if is_adjacent_to_tile:
                    anchors.add((r, c))
        return list(anchors)

    def _gen(self, rack, node, placed, board, r, c, is_horizontal, moves, original_rack):
        """
        Main move generation method, called for each anchor (r, c).
        This corrected version handles both forward-only and backward-then-forward moves.
        """
        # Safety check
        if not (0 <= r < board.size and 0 <= c < board.size):
            return

        # Case 1: Generate moves that only extend forward from the anchor.
        # This corresponds to GADDAG paths that start with the BREAK_CHAR (e.g., '+WORD').
        if self.BREAK_CHAR in node.children:
            self._extend_forward(rack, node.children[self.BREAK_CHAR], [], (r, c), board, is_horizontal, moves, original_rack)
        
        # Case 2: Generate moves that extend backward from the anchor, then pivot and extend forward.
        # The current position and the anchor position start as the same.
        self._extend_backward(rack, node, [], (r, c), (r, c), board, is_horizontal, moves, original_rack)

    def _extend_backward(self, rack, node, placed, current_pos, anchor_pos, board, is_horizontal, moves, original_rack):
        """
        Extends moves backward from the anchor, preserving the anchor's position.
        (This is the corrected version of the original _extend_from_position method).
        """
        r, c = current_pos
        
        # If we find the break character, pivot and start extending forward FROM THE ANCHOR.
        if self.BREAK_CHAR in node.children:
            # This is the critical fix: the forward pass starts from the original `anchor_pos`.
            self._extend_forward(rack, node.children[self.BREAK_CHAR], placed, anchor_pos, board, is_horizontal, moves, original_rack)
        
        # Try placing a tile from the rack at the current backward position
        if board.is_empty(r, c):
            for i in range(len(rack)):
                letter = rack[i]
                if letter == '_':
                    # Handle blank tiles
                    for char_code in range(ord('A'), ord('Z') + 1):
                        char = chr(char_code)
                        if char in node.children:
                            new_rack = rack[:i] + rack[i+1:]
                            new_placed = placed + [(char, (r, c))]
                            next_pos = (r, c - 1) if is_horizontal else (r - 1, c)
                            if 0 <= next_pos[0] < board.size and 0 <= next_pos[1] < board.size:
                                self._extend_backward(new_rack, node.children[char], new_placed, next_pos, anchor_pos, board, is_horizontal, moves, original_rack)
                else:
                    if letter in node.children:
                        new_rack = rack[:i] + rack[i+1:]
                        new_placed = placed + [(letter, (r, c))]
                        next_pos = (r, c - 1) if is_horizontal else (r - 1, c)
                        if 0 <= next_pos[0] < board.size and 0 <= next_pos[1] < board.size:
                            self._extend_backward(new_rack, node.children[letter], new_placed, next_pos, anchor_pos, board, is_horizontal, moves, original_rack)
        
        # If there's an existing tile on the board, use it to continue building backward
        elif not board.is_empty(r, c):
            existing_tile = board.get_tile(r, c)
            if existing_tile and existing_tile.letter in node.children:
                next_pos = (r, c - 1) if is_horizontal else (r - 1, c)
                if 0 <= next_pos[0] < board.size and 0 <= next_pos[1] < board.size:
                    self._extend_backward(rack, node.children[existing_tile.letter], placed, next_pos, anchor_pos, board, is_horizontal, moves, original_rack)

        
        # If there's already a tile on the board, use it
        elif not board.is_empty(r, c):
            existing_tile = board.get_tile(r, c)
            if existing_tile and existing_tile.letter in node.children:
                next_pos = (r, c - 1) if is_horizontal else (r - 1, c)
                # Bounds check before recursion
                if 0 <= next_pos[0] < board.size and 0 <= next_pos[1] < board.size:
                    self._extend_from_position(rack, node.children[existing_tile.letter], placed, next_pos, board, is_horizontal, moves, original_rack)

    def _extend_forward(self, rack, node, placed, pos, board, is_horizontal, moves, original_rack):
        """Extends moves forward from the break character position."""
        r, c = pos
        
        # Check if current placement forms a valid word
        if node.is_terminal and placed:
            self._try_add_move(placed, board, is_horizontal, moves, original_rack)
        
        # Safety check for board boundaries
        if not (0 <= r < board.size and 0 <= c < board.size):
            return
        
        # If current position is empty, try placing tiles
        if board.is_empty(r, c):
            # Use range instead of enumerate to avoid modification during iteration
            for i in range(len(rack)):
                letter = rack[i]
                if letter == '_':
                    # Handle blank tiles
                    for char_code in range(ord('A'), ord('Z') + 1):
                        char = chr(char_code)
                        if char in node.children:
                            new_rack = rack[:i] + rack[i+1:]
                            new_placed = placed + [(char, (r, c))]
                            next_pos = (r, c + 1) if is_horizontal else (r + 1, c)
                            self._extend_forward(new_rack, node.children[char], new_placed, next_pos, board, is_horizontal, moves, original_rack)
                else:
                    if letter in node.children:
                        new_rack = rack[:i] + rack[i+1:]
                        new_placed = placed + [(letter, (r, c))]
                        next_pos = (r, c + 1) if is_horizontal else (r + 1, c)
                        self._extend_forward(new_rack, node.children[letter], new_placed, next_pos, board, is_horizontal, moves, original_rack)
        
        # If there's a tile on the board, use it to continue
        elif not board.is_empty(r, c):
            existing_tile = board.get_tile(r, c)
            if existing_tile and existing_tile.letter in node.children:
                next_pos = (r, c + 1) if is_horizontal else (r + 1, c)
                self._extend_forward(rack, node.children[existing_tile.letter], placed, next_pos, board, is_horizontal, moves, original_rack)


    def _try_add_move(self, placed_tiles, board, is_horizontal, moves, original_rack):
        """Attempts to add a valid move to the moves set with early rack validation."""
        
        try:
            if not placed_tiles:
                return
            
            # EARLY VALIDATION: Check rack compatibility first
            from collections import Counter
            rack_counts = Counter(original_rack)
            placed_letters = [p[0] for p in placed_tiles]
            placed_counts = Counter(placed_letters)
            
            # Check if we have enough of each letter (accounting for blanks)
            blanks_available = rack_counts.get('_', 0)
            blanks_needed = 0
            
            for letter, count in placed_counts.items():
                available = rack_counts.get(letter, 0)
                if available < count:
                    blanks_needed += count - available
            
            if blanks_needed > blanks_available:
                return  # Invalid move - not enough tiles in rack
            
            # Validate cross words
            if not self._is_valid_cross_word_placement(placed_tiles, board, is_horizontal):
                return

            # Construct the full word and determine boundaries
            all_coords = {p[1] for p in placed_tiles}
            
            if is_horizontal:
                r = placed_tiles[0][1][0]
                min_c = min(p[1][1] for p in placed_tiles)
                max_c = max(p[1][1] for p in placed_tiles)
                
                # Extend to find the complete word
                while min_c > 0 and not board.is_empty(r, min_c - 1): 
                    min_c -= 1
                while max_c < board.size - 1 and not board.is_empty(r, max_c + 1): 
                    max_c += 1
                
                # Build the word string
                word = ""
                for c in range(min_c, max_c + 1):
                    if (r, c) in all_coords:
                        word += next(p[0] for p in placed_tiles if p[1] == (r, c))
                    else:
                        tile = board.get_tile(r, c)
                        word += tile.letter if tile else "?"
                
                start_pos, orientation = (r, min_c), 'H'
            else: # Vertical
                c = placed_tiles[0][1][1]
                min_r = min(p[1][0] for p in placed_tiles)
                max_r = max(p[1][0] for p in placed_tiles)
                
                # Extend to find the complete word
                while min_r > 0 and not board.is_empty(min_r - 1, c): 
                    min_r -= 1
                while max_r < board.size - 1 and not board.is_empty(max_r + 1, c): 
                    max_r += 1
                
                # Build the word string
                word = ""
                for r in range(min_r, max_r + 1):
                    if (r, c) in all_coords:
                        word += next(p[0] for p in placed_tiles if p[1] == (r, c))
                    else:
                        tile = board.get_tile(r, c)
                        word += tile.letter if tile else "?"
                
                start_pos, orientation = (min_r, c), 'V'

            # Final validation and addition
            if len(word) > 1 and word in self.dictionary and '?' not in word:
                # Double-check the move can be made with the original rack
                if self._can_make_move_with_rack((word, start_pos, orientation), original_rack, board):
                    moves.add((word, start_pos, orientation))
                
        except Exception:
            # Silently skip invalid moves during generation
            pass

    def _is_valid_cross_word_placement(self, placed_tiles, board, is_horizontal):
        """Validates that all cross-words formed are valid."""
        try:
            for letter, (r, c) in placed_tiles:
                prefix, suffix = "", ""
                
                if is_horizontal: # Check vertical cross-word
                    # Scan up
                    tr = r - 1
                    while tr >= 0 and not board.is_empty(tr, c):
                        tile = board.get_tile(tr, c)
                        prefix = (tile.letter if tile else "") + prefix
                        tr -= 1
                    # Scan down
                    tr = r + 1
                    while tr < board.size and not board.is_empty(tr, c):
                        tile = board.get_tile(tr, c)
                        suffix += tile.letter if tile else ""
                        tr += 1
                else: # Check horizontal cross-word
                    # Scan left
                    tc = c - 1
                    while tc >= 0 and not board.is_empty(r, tc):
                        tile = board.get_tile(r, tc)
                        prefix = (tile.letter if tile else "") + prefix
                        tc -= 1
                    # Scan right
                    tc = c + 1
                    while tc < board.size and not board.is_empty(r, tc):
                        tile = board.get_tile(r, tc)
                        suffix += tile.letter if tile else ""
                        tc += 1
                
                # If there's a cross-word, validate it
                if prefix or suffix:
                    cross_word = prefix + letter + suffix
                    if cross_word not in self.dictionary:
                        return False
            return True
        except Exception:
            return False

    # Legacy method names for compatibility
    def _generate_moves_from_anchor(self, rack, board, r, c, moves):
        """Legacy method for backward compatibility."""
        rack_str = "".join([tile.letter for tile in rack])
        self._gen(rack_str, self.root, [], board, r, c, True, moves)
        self._gen(rack_str, self.root, [], board, r, c, False, moves)

    def _extend_right(self, rack, node, placed, pos, board, moves, is_horizontal):
        """Legacy method for backward compatibility."""
        self._extend_forward(rack, node, placed, pos, board, is_horizontal, moves)

    def _extend_left(self, rack, node, placed, anchor_pos, board, moves, is_horizontal):
        """Legacy method for backward compatibility."""
        # This is a simplified version - the full implementation would be more complex
        pass

    def _validate_and_add_move(self, placed_tiles, board, is_horizontal, moves):
        """Legacy method for backward compatibility."""
        self._try_add_move(placed_tiles, board, is_horizontal, moves)
        
    def debug_rack_usage(self, rack_str, placed_tiles):
        """Debug method to check rack usage."""
        from collections import Counter
        rack_counts = Counter(rack_str)
        placed_letters = [p[0] for p in placed_tiles]
        placed_counts = Counter(placed_letters)
        
        print(f"Rack: {rack_str}")
        print(f"Placed: {placed_letters}")
        print(f"Rack counts: {dict(rack_counts)}")
        print(f"Placed counts: {dict(placed_counts)}")
        
        blanks_available = rack_counts.get('_', 0)
        blanks_needed = 0
        
        for letter, count in placed_counts.items():
            available = rack_counts.get(letter, 0)
            if available < count:
                blanks_needed += count - available
                print(f"Need {count - available} blanks for letter '{letter}'")
        
        print(f"Blanks available: {blanks_available}, blanks needed: {blanks_needed}")
        return blanks_needed <= blanks_available