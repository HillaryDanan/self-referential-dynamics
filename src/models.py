"""
Neural network models for self-referential dynamics experiments.

Key insight: TRUE self-reference means predicting your own INTERNAL STATES,
not just external outputs. This creates genuine recursive self-modeling.

References:
- Shannon (1948). A mathematical theory of communication.
- Jaynes (1957). Information theory and statistical mechanics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional


class BaseNetwork(nn.Module):
    """Base classification network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)


class SelfReferentialNetwork(nn.Module):
    """
    Network with GENUINE self-reference: predicts its own hidden states.
    
    The network learns to:
    1. Classify inputs (primary task)
    2. Predict what its hidden state WILL BE for the current input
       based on what its hidden state WAS for the previous input
    
    This creates recursive self-modeling: the system models its own dynamics.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Main network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # SELF-MODEL: Predicts next hidden state from current hidden state
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden_dim = hidden_dim
        self._prev_hidden: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
        - output: Classification logits
        - predicted_hidden: Self-model's prediction (or None if first batch)
        - actual_hidden: Actual current hidden state
        """
        # Compute current hidden state
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        actual_hidden = h2.detach().clone()
        
        # Output
        output = self.fc3(h2)
        
        # Self-model prediction (only if we have previous hidden state with matching size)
        predicted_hidden = None
        if self._prev_hidden is not None and self._prev_hidden.shape[0] == x.shape[0]:
            predicted_hidden = self.self_model(self._prev_hidden)
        
        # Store current hidden for next iteration
        self._prev_hidden = actual_hidden
        
        return output, predicted_hidden, actual_hidden
    
    def reset_hidden(self):
        """Reset hidden state tracking."""
        self._prev_hidden = None


class NonSelfReferentialNetwork(nn.Module):
    """
    Control network: Same architecture but predicts EXTERNAL targets.
    
    The auxiliary head predicts a transformation of the INPUT,
    not the network's own internal states. This is NOT self-referential.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # Main network (identical to self-referential)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # EXTERNAL MODEL: Reconstructs input (autoencoder-style, NOT self-referential)
        self.external_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - output: Classification logits
        - predicted_input: Reconstruction of input
        - actual_input: The actual input
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        
        output = self.fc3(h2)
        predicted_input = self.external_model(h2)
        
        return output, predicted_input, x


def compute_self_referential_loss(
    classification_loss: torch.Tensor,
    predicted_hidden: Optional[torch.Tensor],
    actual_hidden: torch.Tensor,
    self_weight: float = 0.3
) -> Tuple[torch.Tensor, float]:
    """
    Combined loss for self-referential network.
    
    Returns: (total_loss, self_prediction_error)
    """
    if predicted_hidden is None:
        # First batch - no self-referential signal yet
        return classification_loss, 0.0
    
    self_prediction_loss = F.mse_loss(predicted_hidden, actual_hidden)
    total_loss = classification_loss + self_weight * self_prediction_loss
    
    return total_loss, self_prediction_loss.item()


def compute_external_loss(
    classification_loss: torch.Tensor,
    predicted_external: torch.Tensor,
    actual_external: torch.Tensor,
    external_weight: float = 0.3
) -> Tuple[torch.Tensor, float]:
    """
    Combined loss for non-self-referential network.
    """
    external_loss = F.mse_loss(predicted_external, actual_external)
    total_loss = classification_loss + external_weight * external_loss
    
    return total_loss, external_loss.item()