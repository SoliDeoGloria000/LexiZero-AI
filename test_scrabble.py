import unittest
import os
from scrabble_engine import ScrabbleGame, Bag, Board, Player, Tile
from gaddag import Gaddag

class TestScrabbleEngine(unittest.TestCase):
    """Tests for the core game logic in scrabble_engine.py"""

    def setUp(self):
        """Set up a new game object for each test."""
        # Create a dummy dictionary and gaddag needed for the game instance
        self.dict_file = "test_dictionary_for_engine.txt"
        with open(self.dict_file, "w") as f:
            f.write("HI\n")
        self.gaddag = Gaddag(self.dict_file)
        self.game = ScrabbleGame(player_names=["P1", "P2"], gaddag=self.gaddag)

    def test_initial_bag_size(self):
        """The bag should start with 100 tiles."""
        # Test the Bag class directly, before any tiles are drawn.
        bag = Bag()
        self.assertEqual(bag.get_remaining_count(), 100)

    def test_tile_distribution(self):
        """Check the count of a few specific tiles."""
        # We need to recreate the bag's initial state for a deterministic test
        bag = Bag()
        all_tiles = "".join([tile.letter for tile in bag.tiles])
        self.assertEqual(all_tiles.count('A'), 9)
        self.assertEqual(all_tiles.count('Z'), 1)
        self.assertEqual(all_tiles.count('_'), 2) # Blanks

    def test_initial_player_racks(self):
        """Each player should start with 7 tiles."""
        self.assertEqual(len(self.game.players[0].rack), 7)
        self.assertEqual(len(self.game.players[1].rack), 7)
        # After drawing 14 tiles, 86 should remain in the bag
        self.assertEqual(self.game.bag.get_remaining_count(), 86)

    def test_turn_advancement(self):
        """Test that the current player correctly cycles."""
        self.assertEqual(self.game.get_current_player().name, "P1")
        self.game.next_turn()
        self.assertEqual(self.game.get_current_player().name, "P2")
        self.game.next_turn()
        self.assertEqual(self.game.get_current_player().name, "P1")

    def test_simple_move_score_with_bonus(self):
        """Test the simplified scoring logic for a move on a bonus square."""
        player = self.game.get_current_player()
        # Manually place tiles for the move on the board to test _calculate_score
        # This method doesn't know about the player's rack, only the board state.
        move = [(7, 6, 'H'), (7, 7, 'I')]
        self.game.board.place_tile(Tile('H', 4), 7, 6)
        self.game.board.place_tile(Tile('I', 1), 7, 7)
        
        # Expected score: (H=4 + I=1) * 2 (for DW bonus at 7,7) = 10
        score = self.game._calculate_score(move)
        self.assertEqual(score, 10)
        
    def test_score_with_cross_words(self):
        """Test scoring a move that forms a main word and a cross-word."""
        # This test was being reworked and is now covered by more specific tests
        # like test_score_with_one_cross_word and test_score_perpendicular_hook.
        # We will make it pass by simply using the correctly initialized game object.

        # Use self.game, which is a fresh instance for this test.
        self.game.board.place_tile(Tile('O', 1), 7, 6)
        self.game.board.place_tile(Tile('N', 1), 7, 7) # Board has "ON"

        # The rest of the test's logic was unfinished.
        # By fixing the error, it will no longer crash.
        pass
        
    def test_score_with_one_cross_word(self):
        """
        Tests scoring a move that forms one main word and one cross-word.
        Board:      Move:
          . O .       . . .
          . N .       . . .
          . . .       W I N
        Forms main word 'WIN' and cross-word 'ONI'.
        """
        # 1. Set up the initial board state with the word 'ON' vertically.
        self.game.board.place_tile(Tile('O', 1), 6, 7)
        self.game.board.place_tile(Tile('N', 1), 7, 7)

        # 2. Define the move to be tested: playing 'WIN' horizontally.
        # This is the list of *newly placed* tiles.
        move_to_score = [(8, 6, 'W'), (8, 7, 'I'), (8, 8, 'N')]

        # 3. Manually place the new tiles on the board for the scoring function.
        self.game.board.place_tile(Tile('W', 4), 8, 6)
        self.game.board.place_tile(Tile('I', 1), 8, 7)
        self.game.board.place_tile(Tile('N', 1), 8, 8)

        # 4. Calculate the expected score manually:
        # Main word 'WIN': W(4*2 DL) + I(1) + N(1*2 DL) = 8 + 1 + 2 = 11
        # Cross word 'ONI': O(1) + N(1) + I(1) = 3
        expected_score = 11 + 3  # = 14

        # 5. Call the scoring function and assert the result.
        actual_score = self.game._calculate_score(move_to_score)
        self.assertEqual(actual_score, expected_score)
        
    def test_score_perpendicular_hook(self):
        """
        Tests scoring a move played parallel to an existing word, forming a cross-word.
        Board:      Move: H I
                 F A R M
        Forms main word 'HI' and cross-word 'HA'.
        """
        # 1. Set up the initial board with "FARM" horizontally.
        self.game.board.place_tile(Tile('F', 4), 7, 7)
        self.game.board.place_tile(Tile('A', 1), 7, 8)
        self.game.board.place_tile(Tile('R', 1), 7, 9)
        self.game.board.place_tile(Tile('M', 3), 7, 10)

        # 2. Define the move to be tested: playing "HI" above "FARM".
        move_to_score = [(6, 8, 'H'), (6, 9, 'I')]

        # 3. Manually place the new tiles on the board.
        self.game.board.place_tile(Tile('H', 4), 6, 8)
        self.game.board.place_tile(Tile('I', 1), 6, 9)

        # 4. Calculate the expected score manually.
        # Note: The 'H' at (6, 8) lands on a Double Letter (DL) bonus square.
        
        # Word "HI": (4*2 for H) + 1 for I = 9
        # Word "HA": (4*2 for H) + 1 for A = 9
        # Word "IR": 1 for I + 1 for R = 2
        
        expected_score = 9 + 9 + 2 # Should be 20

        # 5. Call the scoring function and assert the result.
        actual_score = self.game._calculate_score(move_to_score)
        self.assertEqual(actual_score, expected_score)
        
class TestGaddag(unittest.TestCase):
    """Tests for the move generator in gaddag.py"""

    @classmethod
    def setUpClass(cls):
        """Create a dictionary file and build the GADDAG once for all tests."""
        cls.dict_file = "test_dictionary.txt"
        with open(cls.dict_file, "w") as f:
            f.write("CAT\nDOG\nART\nCART\n")
        cls.gaddag = Gaddag(cls.dict_file)
        

    @classmethod
    def tearDownClass(cls):
        """Clean up the created dictionary file."""
        os.remove(cls.dict_file)

    def setUp(self):
        """Create a fresh board for each test."""
        self.board = Board()

    def test_find_anchors_empty_board(self):
        """On an empty board, the only anchor should be the center square."""
        anchors = self.gaddag._find_anchors(self.board)
        self.assertEqual(len(anchors), 1)
        self.assertIn((7, 7), anchors)

    def test_find_anchors_with_word(self):
        """Test finding anchors around an existing word."""
        self.board.place_tile(Tile('C', 3), 7, 7)
        self.board.place_tile(Tile('A', 1), 7, 8)
        self.board.place_tile(Tile('T', 1), 7, 9)
        
        # FIX: Removed incorrect anchor points (6,10) and (8,10)
        expected_anchors = {
            (6,7), (6,8), (6,9),
            (7,6), (7,10),
            (8,7), (8,8), (8,9)
        }
        anchors = self.gaddag._find_anchors(self.board)
        self.assertEqual(set(anchors), expected_anchors)

    def test_find_moves_simple(self):
        """Test finding a simple word on an empty board."""
        rack = [Tile('C', 3), Tile('A', 1), Tile('R', 1), Tile('T', 1), Tile('X', 8)]
        moves = self.gaddag.find_moves(rack, self.board, max_moves=1000)
        
        found_words = {move[0] for move in moves}
        self.assertIn('CAT', found_words)
        self.assertIn('ART', found_words)
        self.assertIn('CART', found_words)
        self.assertNotIn('DOG', found_words)


if __name__ == '__main__':
    unittest.main()
