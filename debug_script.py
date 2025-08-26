"""
Debug script to find exactly where the extra point is coming from.
"""

from scrabble_engine import ScrabbleGame, Tile

def debug_calculate_score_detailed(game, move):
    """
    A detailed debug version of _calculate_score to trace every operation.
    """
    if not move:
        return 0

    print(f"DEBUG: Starting _calculate_score with move: {move}")
    placed_coords = {(r, c) for r, c, _ in move}
    print(f"DEBUG: Placed coordinates: {placed_coords}")
    
    scored_words = set()
    total_score = 0
    
    # Process each newly placed tile
    for i, (r_placed, c_placed, letter) in enumerate(move):
        print(f"\nDEBUG: Processing tile {i+1}/{len(move)}: {letter} at ({r_placed}, {c_placed})")
        
        # Find horizontal word
        start_col = c_placed
        while start_col > 0 and game.board.get_tile(r_placed, start_col - 1) is not None:
            start_col -= 1
        
        end_col = c_placed
        while end_col < game.board.size - 1 and game.board.get_tile(r_placed, end_col + 1) is not None:
            end_col += 1
        
        print(f"  Horizontal span: columns {start_col} to {end_col} (length {end_col - start_col + 1})")
        
        if end_col > start_col:  # Multi-letter word
            word_key = ('H', r_placed, start_col, end_col)
            print(f"  Horizontal word key: {word_key}")
            
            if word_key not in scored_words:
                word_coords = [(r_placed, c) for c in range(start_col, end_col + 1)]
                word_letters = []
                for r, c in word_coords:
                    tile = game.board.get_tile(r, c)
                    word_letters.append(tile.letter if tile else '?')
                word_str = ''.join(word_letters)
                
                word_score = debug_score_word(game, word_coords, placed_coords, f"horizontal '{word_str}'")
                total_score += word_score
                scored_words.add(word_key)
                print(f"  Added {word_score} points for horizontal word '{word_str}'")
            else:
                print(f"  Skipped horizontal word (already scored): {word_key}")
        else:
            print(f"  No horizontal multi-letter word (single letter)")
        
        # Find vertical word
        start_row = r_placed
        while start_row > 0 and game.board.get_tile(start_row - 1, c_placed) is not None:
            start_row -= 1
        
        end_row = r_placed
        while end_row < game.board.size - 1 and game.board.get_tile(end_row + 1, c_placed) is not None:
            end_row += 1
        
        print(f"  Vertical span: rows {start_row} to {end_row} (length {end_row - start_row + 1})")
        
        if end_row > start_row:  # Multi-letter word
            word_key = ('V', c_placed, start_row, end_row)
            print(f"  Vertical word key: {word_key}")
            
            if word_key not in scored_words:
                word_coords = [(r, c_placed) for r in range(start_row, end_row + 1)]
                word_letters = []
                for r, c in word_coords:
                    tile = game.board.get_tile(r, c)
                    word_letters.append(tile.letter if tile else '?')
                word_str = ''.join(word_letters)
                
                word_score = debug_score_word(game, word_coords, placed_coords, f"vertical '{word_str}'")
                total_score += word_score
                scored_words.add(word_key)
                print(f"  Added {word_score} points for vertical word '{word_str}'")
            else:
                print(f"  Skipped vertical word (already scored): {word_key}")
        else:
            print(f"  No vertical multi-letter word (single letter)")
    
    print(f"\nDEBUG: All words scored: {scored_words}")
    print(f"DEBUG: Total score before bingo bonus: {total_score}")
    
    # Bingo bonus
    if len(placed_coords) == 7:
        total_score += 50
        print("DEBUG: Added 50-point bingo bonus")
    
    print(f"DEBUG: Final total score: {total_score}")
    return total_score

def debug_score_word(game, word_coords, placed_coords, word_description):
    """Debug version of _score_word."""
    print(f"    Scoring {word_description} at {word_coords}")
    
    letter_score = 0
    word_multiplier = 1
    
    for r, c in word_coords:
        tile = game.board.get_tile(r, c)
        if tile is None:
            print(f"      ERROR: No tile at ({r},{c})")
            continue
            
        tile_value = tile.value
        original_value = tile_value
        
        # Apply bonuses only to newly placed tiles
        if (r, c) in placed_coords:
            bonus = game.board.bonus_squares.get((r, c))
            if bonus == "DL":
                tile_value *= 2
                print(f"      {tile.letter} at ({r},{c}): {original_value} -> {tile_value} (DL bonus)")
            elif bonus == "TL":
                tile_value *= 3
                print(f"      {tile.letter} at ({r},{c}): {original_value} -> {tile_value} (TL bonus)")
            elif bonus == "DW":
                word_multiplier *= 2
                print(f"      {tile.letter} at ({r},{c}): {tile_value}, word multiplier now {word_multiplier} (DW)")
            elif bonus == "TW":
                word_multiplier *= 3
                print(f"      {tile.letter} at ({r},{c}): {tile_value}, word multiplier now {word_multiplier} (TW)")
            else:
                print(f"      {tile.letter} at ({r},{c}): {tile_value} (newly placed, no bonus)")
        else:
            print(f"      {tile.letter} at ({r},{c}): {tile_value} (existing tile)")
        
        letter_score += tile_value
    
    final_score = letter_score * word_multiplier
    print(f"    {word_description}: {letter_score} x {word_multiplier} = {final_score}")
    return final_score

def main():
    """Run the detailed debugging."""
    game = ScrabbleGame(player_names=["P1", "P2"])
    
    # Set up the board
    game.board.place_tile(Tile('O', 1), 6, 7)
    game.board.place_tile(Tile('N', 1), 7, 7)
    game.board.place_tile(Tile('W', 4), 8, 6)
    game.board.place_tile(Tile('I', 1), 8, 7)
    game.board.place_tile(Tile('N', 1), 8, 8)
    
    move_to_score = [(8, 6, 'W'), (8, 7, 'I'), (8, 8, 'N')]
    
    print("=== DETAILED DEBUG TRACE ===")
    debug_score = debug_calculate_score_detailed(game, move_to_score)
    
    print(f"\n=== COMPARISON ===")
    print(f"Debug trace result: {debug_score}")
    print(f"Actual function result: {game._calculate_score(move_to_score)}")
    print(f"Expected result: 13")

if __name__ == "__main__":
    main()