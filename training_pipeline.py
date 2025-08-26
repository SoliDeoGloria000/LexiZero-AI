import torch
import torch.optim as optim
from collections import deque
import random
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from tqdm import tqdm
import os
import time

# --- Project Imports ---
from scrabble_engine import ScrabbleGame, BOARD_SIZE
from gaddag import Gaddag
from lexizero_net import LexiZeroNet
from state_encoder import encode_state
from pomcts import pomcts_search

# Global variable to hold the GADDAG instance for each worker
worker_gaddag = None

def init_worker(dictionary_path):
    """
    Initializer for each worker process in the pool.
    Creates a single GADDAG instance for that worker.
    """
    global worker_gaddag
    print(f"Initializing worker process {os.getpid()}...")
    worker_gaddag = Gaddag(dictionary_path)
    print(f"Worker {os.getpid()} initialized.")

# --- Constants for Training ---
# Force CPU for multiprocessing to avoid CUDA context issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPLAY_BUFFER_SIZE = 250000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_SIMULATIONS = 20  # Reduced for faster debugging
TURNS_UNTIL_GREEDY = 10
NUM_EVAL_GAMES = 20
WIN_RATE_THRESHOLD = 0.55
NUM_WORKERS = 2  # Reduced for debugging
MAX_GAME_TURNS = 100  # Add maximum turns to prevent infinite games

class ReplayBuffer:
    """
    A simple FIFO buffer to store training examples from self-play.
    """
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        """Adds a new experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Samples a batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def run_self_play_game(network_weights):
    """
    Plays one full game of Scrabble against itself, generating training data.
    """
    try:
        network = LexiZeroNet()
        network.load_state_dict(network_weights)
        network.eval()

        game = ScrabbleGame(player_names=["LexiZero_A", "LexiZero_B"], gaddag=worker_gaddag)
        game_history = []

        start_time = time.time()
        TIMEOUT_SECONDS = 60

        # This loop now correctly wraps the game logic
        for turn_count in range(1, MAX_GAME_TURNS + 1):
            elapsed = time.time() - start_time
            print(f"Worker {os.getpid()}: Turn {turn_count}, elapsed {elapsed:.1f}s")

            if elapsed > TIMEOUT_SECONDS:
                print(f"Worker {os.getpid()}: Timeout reached, ending game early.")
                break

            if game.is_game_over():
                break

            current_player_perspective = game.current_player_index
            temperature = 1.0 if game.turn_number <= TURNS_UNTIL_GREEDY else 0.0

            try:
                # Use a CPU device context for workers
                policy = pomcts_search(game, network, worker_gaddag, NUM_SIMULATIONS, temperature, torch.device('cpu'))
            except Exception as e:
                print(f"Worker {os.getpid()}: POMCTS search failed: {e}")
                policy = {}
            
            if not policy:
                # --- ADD THIS LOGGING BLOCK ---
                player_name = game.get_current_player().name
                rack_str = "".join(sorted([t.letter for t in game.get_current_player().rack]))
                print(
                    f"Worker {os.getpid()}: Turn {turn_count} | Player: {player_name} | "
                    f"Rack: [{rack_str}] | No suitable moves found. Passing turn."
                )
                # --- END OF ADDED BLOCK ---
                game.pass_turn()
                continue

            # --- Store state BEFORE the move is made ---
            board_tensor, scalar_vector = encode_state(game)
            game_history.append({
                "board_tensor": board_tensor,
                "scalar_vector": scalar_vector,
                "policy_target": policy,
                "player_perspective": current_player_perspective
            })

            moves = list(policy.keys())
            probabilities = np.array(list(policy.values()), dtype=np.float32)
            probabilities /= probabilities.sum() # Ensure normalization
            chosen_index = np.random.choice(len(moves), p=probabilities)
            move_to_play = moves[chosen_index]
            
            # --- ADD THIS LOGGING BLOCK ---
            player = game.get_current_player()
            rack_str = "".join(sorted([t.letter for t in player.rack]))
            print(
                f"Worker {os.getpid()}: Turn {turn_count} | Player: {player.name} | "
                f"Rack: [{rack_str}] | Chosen Move: {move_to_play} | "
                f"Score: {player.score}"
            )
            

            try:
                game.play_move(move_to_play)
            except Exception as e:
                print(f"Worker {os.getpid()}: Move '{move_to_play}' failed: {e}, passing.")
                game.pass_turn()

        # --- Post-game processing ---
        p0_outcome = game.get_outcome()

        training_examples = []
        for experience in game_history:
            value_target = p0_outcome if experience["player_perspective"] == 0 else -p0_outcome
            training_examples.append((
                experience["board_tensor"],
                experience["scalar_vector"],
                experience["policy_target"],
                value_target
            ))
        
        return training_examples
        
    except Exception as e:
        print(f"Worker {os.getpid()}: Self-play game failed with error: {e}")
        import traceback
        traceback.print_exc()
        return []
    
def train_network(network, optimizer, replay_buffer):
    """
    Samples a batch from the replay buffer and trains the network for one step.
    """
    if len(replay_buffer) < BATCH_SIZE:
        return None

    # 1. Sample a batch of experiences
    batch = replay_buffer.sample(BATCH_SIZE)
    
    # Unzip the batch
    board_tensors, scalar_vectors, policy_targets, value_targets = zip(*batch)

    # Convert lists to tensors for PyTorch
    board_batch = torch.from_numpy(np.array(board_tensors)).to(device)
    scalar_batch = torch.from_numpy(np.array(scalar_vectors)).to(device)
    value_targets = torch.tensor(value_targets, dtype=torch.float32).view(-1, 1).to(device)

    network.train()
    
    # VALUE PASS
    _, value_preds = network(board_batch, scalar_batch)
    value_loss = torch.nn.functional.mse_loss(value_preds, value_targets)

    # POLICY PASS
    policy_input_boards = []
    policy_input_scalars = []
    policy_target_vectors = []
    
    for i in range(BATCH_SIZE):
        base_board_tensor = board_tensors[i]
        scalar_vector = scalar_vectors[i]
        policy_target_dict = policy_targets[i]
        
        moves = list(policy_target_dict.keys())
        move_probs = torch.tensor(list(policy_target_dict.values()), dtype=torch.float32)
        policy_target_vectors.append(move_probs)

        # Create an input for each legal move
        for move in moves:
            # The move is encoded in the 'current move' plane (channel 84)
            move_plane = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
            word, (start_r, start_c), orientation = move
            r, c = start_r, start_c
            for _ in word:
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    move_plane[0, r, c] = 1.0
                r, c = (r, c + 1) if orientation == 'H' else (r + 1, c)

            # Combine the base state with the move evaluation plane
            combined_board = np.copy(base_board_tensor)
            combined_board[84, :, :] = move_plane

            policy_input_boards.append(combined_board)
            policy_input_scalars.append(scalar_vector)
    
    # Convert the list of all moves into a single batch
    policy_board_batch = torch.from_numpy(np.array(policy_input_boards)).to(device)
    policy_scalar_batch = torch.from_numpy(np.array(policy_input_scalars)).to(device)

    # Perform the forward pass to get policy logits
    policy_logits, _ = network(policy_board_batch, policy_scalar_batch)
    
    # Group the logits by the original game state
    grouped_logits = []
    start_idx = 0
    for target in policy_target_vectors:
        num_moves = len(target)
        grouped_logits.append(policy_logits[start_idx : start_idx + num_moves].view(-1))
        start_idx += num_moves

    # Calculate Policy Loss: Cross-Entropy between network policy and MCTS policy
    policy_loss = 0
    for i in range(BATCH_SIZE):
        # Apply softmax to the logits before calculating cross-entropy with probabilities
        log_probs = torch.nn.functional.log_softmax(grouped_logits[i], dim=0)
        target_probs_on_device = policy_target_vectors[i].to(device)
        policy_loss += -torch.sum(target_probs_on_device * log_probs)

    policy_loss /= BATCH_SIZE

    # COMBINE LOSS AND BACKPROPAGATE
    total_loss = value_loss + policy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), value_loss.item(), policy_loss.item()
    
def run_evaluation_game(p1_net_weights, p2_net_weights, p1_name, p2_name):
    """
    Plays a single game between two networks and returns the winner's name.
    FIXED: Removed device parameter and added safety checks.
    """
    try:
        p1_net = LexiZeroNet()
        p1_net.load_state_dict(p1_net_weights)
        p1_net.eval()

        p2_net = LexiZeroNet()
        p2_net.load_state_dict(p2_net_weights)
        p2_net.eval()
        
        # Use the global 'worker_gaddag' initialized for this process
        game = ScrabbleGame(player_names=[p1_name, p2_name], gaddag=worker_gaddag)
        
        turn_count = 0
        while not game.is_game_over() and turn_count < MAX_GAME_TURNS:
            turn_count += 1
            current_net = p1_net if game.current_player_index == 0 else p2_net
            
            try:
                policy = pomcts_search(game, current_net, worker_gaddag, NUM_SIMULATIONS, temperature=0, device=torch.device('cpu'))
            except Exception as e:
                print(f"Evaluation game POMCTS failed: {e}")
                policy = {}
            
            if not policy:
                game.pass_turn()
                continue
            
            move_to_play = max(policy, key=policy.get)
            try:
                game.play_move(move_to_play)
            except Exception as e:
                print(f"Evaluation move failed: {e}")
                game.pass_turn()

        outcome = game.get_outcome()
        
        if outcome == 1.0:
            return game.players[0].name
        elif outcome == -1.0:
            return game.players[1].name
        else:
            return "Draw"
            
    except Exception as e:
        print(f"Evaluation game failed: {e}")
        return "Draw"  # Default to draw on error
    
def evaluate_networks(challenger_net, best_net, gaddag, device):
    """
    Pits the challenger network against the best network in a parallelized match.
    """
    print("Starting network evaluation...")
    
    # Get the state dicts from the models (on CPU for safe pickling)
    challenger_net.to('cpu')
    best_net.to('cpu')
    challenger_weights = challenger_net.state_dict()
    best_weights = best_net.state_dict()
    
    # Move models back to device for the main process
    challenger_net.to(device)
    best_net.to(device)

    worker_args = []
    # Create arguments for NUM_EVAL_GAMES, alternating the starting player
    for i in range(NUM_EVAL_GAMES):
        if i % 2 == 0:
            # Best net starts as Player 1
            p1_weights, p2_weights = best_weights, challenger_weights
            p1_name, p2_name = "Best", "Challenger"
        else:
            # Challenger starts as Player 1
            p1_weights, p2_weights = challenger_weights, best_weights
            p1_name, p2_name = "Challenger", "Best"
        
        # REMOVED device parameter
        worker_args.append((p1_weights, p2_weights, p1_name, p2_name))

    # Run the evaluation games in parallel
    with mp.Pool(NUM_WORKERS, initializer=init_worker, initargs=(gaddag.dictionary_path,)) as pool:
        results = list(tqdm(pool.starmap(run_evaluation_game, worker_args), total=NUM_EVAL_GAMES, desc="Evaluating Networks"))

    # Count the wins for the challenger
    challenger_wins = results.count("Challenger")
    
    win_rate = challenger_wins / NUM_EVAL_GAMES
    print(f"Evaluation complete. Challenger win rate: {win_rate:.2f} ({challenger_wins}/{NUM_EVAL_GAMES})")
    return win_rate

def main():
    """
    The main training loop orchestrating the entire process.
    """
    print("--- Python script has started ---")
    print(f"--- Using device: {device.type.upper()} ---")
    
    print("Initializing LexiZero Training Pipeline...")
     
    # This will create a 'runs' directory to store the logs
    writer = SummaryWriter()

    # Initialize components
    dictionary_path = "CSW24_words_only.txt"
    gaddag = Gaddag(dictionary_path=dictionary_path)
    
    # The 'best_network' is our champion model used for generating games
    best_network = LexiZeroNet().to(device)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # Check for a checkpoint to resume training
    RESUME_CHECKPOINT_PATH = None
    start_generation = 0
    
    if RESUME_CHECKPOINT_PATH and os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"--- Resuming training from {RESUME_CHECKPOINT_PATH} ---")
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH)
        
        best_network.load_state_dict(checkpoint['model_state_dict'])
        challenger_for_opt = LexiZeroNet().to(device)
        challenger_for_opt.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(challenger_for_opt.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        replay_buffer.buffer.extend(checkpoint['replay_buffer'])
        start_generation = checkpoint['generation']
        
        print(f"Resumed successfully. Starting at generation {start_generation + 1}.")
        print(f"Replay buffer loaded with {len(replay_buffer)} experiences.")
    else:
        print("--- Starting a new training run ---")

    # Main Training Loop
    for generation in range(start_generation, 1):  # Increased from 1 to 100
        print(f"\n--- Generation {generation+1} ---")

        # Phase 1: Generate Games using the current best network (PARALLELIZED)
        print("Generating self-play games with the best network...")
        best_network.eval()

        # Move network to CPU to get weights for safe transfer to workers
        best_network.to('cpu')
        network_weights = best_network.state_dict()
        best_network.to(device)

        GAMES_PER_GENERATION = 1  # Reduced for debugging

        # Prepare the arguments for each worker process (REMOVED device parameter)
        worker_args = [(network_weights,) for _ in range(GAMES_PER_GENERATION)]

        print(f"Starting {GAMES_PER_GENERATION} self-play games with {NUM_WORKERS} workers...")
        start_time = time.time()
        
        with mp.Pool(NUM_WORKERS, initializer=init_worker, initargs=(dictionary_path,)) as pool:
            results = list(tqdm(pool.starmap(run_self_play_game, worker_args), total=GAMES_PER_GENERATION, desc="Self-play games"))

        end_time = time.time()
        print(f"Self-play completed in {end_time - start_time:.2f} seconds")

        # Add the collected game data to the replay buffer
        total_examples = 0
        for game_data in results:
            for example in game_data:
                replay_buffer.add(example)
                total_examples += 1
        
        print(f"  {GAMES_PER_GENERATION} games complete. Generated {total_examples} examples. Replay buffer size: {len(replay_buffer)}")

        # Phase 2: Train a new challenger network
        print("Training challenger network...")
        challenger_network = LexiZeroNet().to(device)
        challenger_network.load_state_dict(best_network.state_dict())
        optimizer = optim.Adam(challenger_network.parameters(), lr=LEARNING_RATE)
        
        if len(replay_buffer) >= BATCH_SIZE:
            global_step = generation * 50
            training_losses = []

            for step in range(50):
                loss = train_network(challenger_network, optimizer, replay_buffer)
                if loss:
                    training_losses.append(loss[0])
                    if (step + 1) % 10 == 0:
                        print(f"  Training step {step+1} complete. Loss: {loss[0]:.4f}")
                        writer.add_scalar('Loss/total', loss[0], global_step + step)
                        writer.add_scalar('Loss/value', loss[1], global_step + step)
                        writer.add_scalar('Loss/policy', loss[2], global_step + step)
            
            avg_loss = np.mean(training_losses) if training_losses else 0
            print(f"  Average training loss: {avg_loss:.4f}")
        else:
            print("  Not enough data in buffer to train. Skipping.")
            continue

        # Phase 3: Evaluate & Update
        win_rate = evaluate_networks(challenger_network, best_network, gaddag, device)
        
        writer.add_scalar('Evaluation/WinRate_vs_Best', win_rate, generation)

        # Phase 4: Checkpointing
        CHECKPOINT_INTERVAL = 10
        if (generation + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f'lexizero_gen_{generation+1}.pth'
            print(f"--- Saving checkpoint to {checkpoint_path} ---")
            
            checkpoint = {
                'generation': generation + 1,
                'model_state_dict': best_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'replay_buffer': list(replay_buffer.buffer), 
            }
            torch.save(checkpoint, checkpoint_path)
            
        # Update best network if challenger wins
        if win_rate > WIN_RATE_THRESHOLD:
            print(f"New best network found! Promoting challenger.")
            best_network = challenger_network
            torch.save(best_network.state_dict(), 'best_lexizero_net.pth') 
        else:
            print("Challenger did not meet threshold. Keeping previous best network.")
            
    print("\nTraining complete.")
    writer.close()
    
if __name__ == "__main__":
    # Import the necessary modules for profiling
    import cProfile
    import pstats

    # Create a profiler object
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your main function
    main()

    profiler.disable()
    
    # Save the profiling results to a file
    print("--- Profiling complete. Saving stats to 'lexizero.prof' ---")
    profiler.dump_stats("lexizero.prof")

    # Optional: To print the 10 slowest functions directly to the console
    # print("--- Top 10 functions by cumulative time ---")
    # stats = pstats.Stats(profiler).sort_stats('cumulative')
    # stats.print_stats(10)
