import torch
import numpy as np
import random
from collections import defaultdict
import logging

# --- Project Imports ---
# Make sure these imports match your project structure
from scrabble_engine import ScrabbleGame, TILE_DISTRIBUTION, Tile
from lexizero_net import LexiZeroNet
from gaddag import Gaddag
from state_encoder import encode_state

# --- Constants ---
C_PUCT = 1.0

logger = logging.getLogger(__name__)

# --- THIS ENTIRE HELPER FUNCTION IS NEW ---
def get_heuristic_score(move, board):
    """Calculates the heuristic score for a single move."""
    # Exchanges are not scored with this heuristic
    if not isinstance(move, tuple) or len(move) != 3:
        return 0

    word, (start_r, start_c), orientation = move
    
    word_score = 0
    word_multiplier = 1
    letters_placed = 0
    
    r, c = start_r, start_c
    for letter in word:
        if board.is_empty(r, c):
            letters_placed += 1
            tile_value = TILE_DISTRIBUTION.get(letter, (0, 0))[1]
            bonus = board.bonus_squares.get((r, c))
            if bonus == "DL": tile_value *= 2
            elif bonus == "TL": tile_value *= 3
            elif bonus == "DW": word_multiplier *= 2
            elif bonus == "TW": word_multiplier *= 3
            word_score += tile_value
        else:
            existing_tile = board.get_tile(r, c)
            word_score += existing_tile.value if existing_tile else 0
        r, c = (r, c + 1) if orientation == 'H' else (r + 1, c)

    total_score = word_score * word_multiplier
    if letters_placed == 7:
        total_score += 50
        
    return total_score


def prune_moves(moves, board, top_k=50):
    """
    Prunes a list of moves by keeping the top_k based on a lightweight scoring heuristic.
    """
    if len(moves) <= top_k:
        return moves

    # Score all moves using the heuristic function
    # Note: Exchange moves ('EXCHANGE', ...) will get a score of 0
    scored_moves = [(move, get_heuristic_score(move, board)) for move in moves]

    # Give exchanges a slight boost to ensure they are considered if rack is poor
    for i, (move, score) in enumerate(scored_moves):
        if isinstance(move, tuple) and move[0] == 'EXCHANGE':
            scored_moves[i] = (move, 1) # Give exchange a score of 1

    # Sort by the heuristic score in descending order
    scored_moves.sort(key=lambda x: x[1], reverse=True)

    # Return only the move tuples from the top candidates
    return [move for move, score in scored_moves[:top_k]]


class Node:
    """A node in the Monte Carlo search tree."""
    def __init__(self, parent=None, prior=0):
        self.parent = parent
        self.children = {}  # map from action to Node
        self.visit_count = 0
        self.q_value = 0
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        """Selects a child node using the PUCT formula."""
        best_score = -np.inf
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.q_value + C_PUCT * child.prior * \
                    (np.sqrt(self.visit_count) / (1 + child.visit_count))
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, legal_moves, priors):
        """Expands the node by creating children for each legal move."""
        for move in legal_moves:
            if move not in self.children:
                self.children[move] = Node(parent=self, prior=priors.get(move, 0))

    def update(self, value):
        """Updates the node's statistics after a simulation."""
        self.visit_count += 1
        self.q_value += (value - self.q_value) / self.visit_count

def sample_determinization(game_state: ScrabbleGame):
        """
        Creates a single, fully-determined 'hypothetical world' from a belief state.
        It samples the opponent's rack and the bag's contents.
    
        Args:
        game_state: The game state object to determinize.
    
        Returns:
        A new ScrabbleGame object with all hidden info resolved.
        """
        # Get all tiles that are not on the board and not in the current player's rack
        unseen_counts = game_state.get_unseen_tile_counts()
        unseen_tiles = []
        for letter, count in unseen_counts.items():
            if count > 0:
                value = TILE_DISTRIBUTION[letter][1]
                unseen_tiles.extend([Tile(letter, value) for _ in range(count)])
    
        random.shuffle(unseen_tiles)

        # Assign tiles to the opponent's rack
        opponent = game_state.get_opponent()
        opponent_rack_size = len(opponent.rack) # Use the known size of the opponent's rack
    
        opponent.rack = unseen_tiles[:opponent_rack_size]
    
        # The rest of the unseen tiles become the bag
        game_state.bag.tiles = unseen_tiles[opponent_rack_size:]
    
        return game_state
        
        
        
def pomcts_search(game: ScrabbleGame, network: LexiZeroNet, gaddag: Gaddag, num_simulations: int, temperature: float, device):
    """
    Performs Partially Observable Monte Carlo Tree Search.
    """
    root = Node()

    for _ in range(num_simulations):
        # Create a deep copy for the simulation to not alter the original game state
        sim_game = game.clone() 
        
        # --- THIS IS THE CRITICAL FIX ---
        # Determinize the game state for THIS simulation run
        sim_game = sample_determinization(sim_game)
        
        node = root
        search_path = [node]

        # --- Step 1: Selection ---
        while not node.is_leaf():
            # Select the next action and corresponding child node
            action, next_node = node.select_child()
            try:
                # Attempt to apply the move to the simulation game state
                sim_game.play_move(action)
            except Exception:
                # If the move is invalid (e.g., uses tiles not in rack),
                # remove this child from consideration and retry selection.
                del node.children[action]
                if not node.children:
                    # No valid moves remain from this node
                    break
                continue
            # Move was successful; advance to the child node
            node = next_node
            search_path.append(node)

        # --- Step 2: Expansion & Evaluation ---
        if sim_game.is_game_over():
            value = sim_game.get_outcome()
        else:
            # Determinize the current state for this simulation's evaluation
            determinized_world = sim_game 
            
            rack = determinized_world.get_current_player().rack
            board = determinized_world.board
            legal_moves = gaddag.find_moves(rack, board)

            if determinized_world.bag.get_remaining_count() >= len(rack):
                rack_letters = "".join(sorted([tile.letter for tile in rack]))
                exchange_move = ('EXCHANGE', rack_letters)
                legal_moves.append(exchange_move)

            # --- THE LOGIC FROM THIS POINT FORWARD IS MODIFIED ---
            pruned_moves = prune_moves(legal_moves, determinized_world.board, top_k=50)
            logger.debug(
                "    [Leaf Expansion] Found %d moves, pruned to %d for NN eval.",
                len(legal_moves),
                len(pruned_moves),
            )


            if not pruned_moves:
                value = 0 # No moves possible, treat as neutral
            else:
                # 1. Create a batch of inputs, one for each *pruned* move.
                # --- OPTIMIZATION: ENCODE BASE TENSOR ONLY ONCE ---
                base_board_tensor, base_scalar_vector = encode_state(determinized_world, move_to_evaluate=None)
                
                board_tensors = []
                for move in pruned_moves:
                    # Create a copy and apply the move plane to avoid re-encoding
                    board_tensor_for_move = np.copy(base_board_tensor)
                    
                    # Only apply move plane for actual scoring moves
                    if isinstance(move, tuple) and len(move) == 3:
                        word, (start_r, start_c), orientation = move
                        r, c = start_r, start_c
                        for _ in word:
                            if 0 <= r < 15 and 0 <= c < 15 and board.is_empty(r, c):
                                board_tensor_for_move[84, r, c] = 1.0
                            r, c = (r, c + 1) if orientation == 'H' else (r + 1, c)

                    board_tensors.append(board_tensor_for_move)
                # --- END OF OPTIMIZATION ---

                # 2. Convert to PyTorch tensors.
                board_batch = torch.from_numpy(np.array(board_tensors)).to(device)
                scalar_batch = torch.from_numpy(np.array([base_scalar_vector] * len(pruned_moves))).to(device)

                # 3. Get network predictions.
                with torch.no_grad():
                    policy_logits, value_batch = network(board_batch, scalar_batch)

                value = value_batch.mean().item() # Use the average value of states as the node's value
                policy_probs = torch.softmax(policy_logits.view(-1), dim=0).cpu().numpy()
                
                priors = {move: prob for move, prob in zip(pruned_moves, policy_probs)}
                
                # Expand the leaf node with the pruned moves and their priors
                search_path[-1].expand(pruned_moves, priors)

        # --- Step 3: Backpropagation ---
        for node in reversed(search_path):
            node.update(value)
            value = -value # The value is from the perspective of the other player

    # After all simulations, determine the policy based on visit counts
    if not root.children:
        return {}
        
    visit_counts = np.array([child.visit_count for child in root.children.values()])
    actions = list(root.children.keys())
    
    if temperature == 0: # Greedy selection
        probs = np.zeros_like(visit_counts, dtype=float)
        probs[np.argmax(visit_counts)] = 1.0
    else:
        # Apply temperature for exploration
        visit_counts_temp = visit_counts**(1/temperature)
        probs = visit_counts_temp / np.sum(visit_counts_temp)

    final_policy = {action: prob for action, prob in zip(actions, probs)}
    
    # --- ADD THIS FINAL CHECK ---
    # If the best move is weak, prefer to pass/exchange to end the game.
    MIN_SCORE_THRESHOLD = 5
    # Check if the game is past the opening phase
    if game.turn_number > 4 and final_policy:
        best_move = max(final_policy, key=final_policy.get)
        best_move_score = get_heuristic_score(best_move, game.board)
        
        # If the best move scores less than the threshold, return nothing.
        # This will cause the agent to pass its turn.
        if best_move_score < MIN_SCORE_THRESHOLD:
            return {} # Force a pass

    return final_policy
