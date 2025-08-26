import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Constants (should match state_encoder.py) ---
BOARD_SIZE = 15
SCALAR_VECTOR_SIZE = 85 # 27 (rack) + 54 (belief) + 4 (metadata) = 85 (approx)

class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers.
    This is the core building block of the network's body.
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # Skip connection
        out = F.relu(out)
        return out

class LexiZeroNet(nn.Module):
    """
    The main dual-head neural network for LexiZero.
    It processes the game state and outputs a policy and a value.
    """
    def __init__(self, num_res_blocks=10, num_channels=128, scalar_features_dim=SCALAR_VECTOR_SIZE):
        super(LexiZeroNet, self).__init__()
        
        # --- Network Body ---
        # Initial convolutional layer to process the 85 input channels
        self.initial_conv = nn.Conv2d(85, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_initial = nn.BatchNorm2d(num_channels)
        
        # Stack of residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])
        
        # --- Policy Head ---
        # The input dimension is the flattened output of the conv body plus the scalar features
        conv_output_flat_dim = num_channels * BOARD_SIZE * BOARD_SIZE
        combined_features_dim = conv_output_flat_dim + scalar_features_dim
        
        self.policy_fc1 = nn.Linear(combined_features_dim, 256)
        # As per the blueprint, the policy head outputs a single logit per move.
        # In a real implementation, this would be more complex, but for the architecture
        # definition, a single output neuron is a valid representation for one move.
        self.policy_fc2 = nn.Linear(256, 1)

        # --- Value Head ---
        self.value_fc1 = nn.Linear(combined_features_dim, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor, scalar_vector):
        """
        Performs a forward pass through the network.

        Args:
            board_tensor (torch.Tensor): The (N, 85, 15, 15) tensor from the state encoder.
            scalar_vector (torch.Tensor): The (N, scalar_features_dim) vector.

        Returns:
            A tuple containing:
            - policy_logits (torch.Tensor): Logits for the policy head.
            - value (torch.Tensor): The estimated value of the state [-1, 1].
        """
        # Pass through the network body
        x = F.relu(self.bn_initial(self.initial_conv(board_tensor)))
        for block in self.res_blocks:
            x = block(x)
        
        # Flatten the output of the convolutional body
        x_flat = x.view(x.size(0), -1)
        
        # Concatenate the flattened board features with the scalar features
        combined_features = torch.cat([x_flat, scalar_vector], dim=1)
        
        # --- Value Head Pass ---
        v = F.relu(self.value_fc1(combined_features))
        value = torch.tanh(self.value_fc2(v)) # tanh activation to constrain output to [-1, 1]
        
        # --- Policy Head Pass ---
        # The blueprint notes that the policy head evaluates a list of candidate moves.
        # For this architectural definition, we'll return a single representative logit.
        # The training loop will be responsible for iterating over legal moves.
        p = F.relu(self.policy_fc1(combined_features))
        policy_logits = self.policy_fc2(p)
        
        return policy_logits, value

# --- Example Usage ---
if __name__ == '__main__':
    # This demonstrates how to create the network and pass dummy data through it
    # to verify the architecture and tensor shapes.
    
    print("Initializing LexiZeroNet...")
    # Create an instance of the network
    net = LexiZeroNet(num_res_blocks=5, num_channels=64) # Smaller version for testing
    print(net)
    
    # Create dummy input tensors with a batch size of 1
    # This simulates the output from our state_encoder.py
    dummy_board_tensor = torch.randn(1, 85, BOARD_SIZE, BOARD_SIZE)
    dummy_scalar_vector = torch.randn(1, SCALAR_VECTOR_SIZE)
    
    print("\nPerforming a forward pass with dummy data...")
    # Put the network in evaluation mode
    net.eval()
    with torch.no_grad(): # Disable gradient calculation for inference
        policy_logits, value = net(dummy_board_tensor, dummy_scalar_vector)
    
    print(f"\nOutput Policy Logits Shape: {policy_logits.shape}") # Expected: (1, 1)
    print(f"Output Value Shape: {value.shape}") # Expected: (1, 1)
    print(f"Example Value Output: {value.item():.4f}")

