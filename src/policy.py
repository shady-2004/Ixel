"""
policy.py

PURPOSE:
  Implements a contextual bandit policy: a small neural network that maps a
  column's 16-dim feature vector to a predicted "value" (Q-value) for each
  possible action, plus the logic to (a) pick an action from those values and
  (b) learn from the reward that comes back after the action is evaluated.

  This file does NOT touch any database. It only deals with plain numbers in,
  plain numbers/strings out. The simulator (a different file) is responsible
  for actually applying an action and computing the reward.

LIBRARIES NEEDED:
  - torch            (pip install torch)
  - torch.nn         (network layers)
  - torch.optim      (the optimizer that updates weights)
  - random           (stdlib -- for epsilon-greedy exploration)
  - typing           (stdlib -- for type hints, optional but good practice)
"""

from torch.nn import ReLU
from torch.nn import Linear
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List



ACTIONS = ["INDEX", "NO_ACTION", "DROP"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTIONS)}


class QNetwork(nn.Module):


    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_actions: int = 3):
        super().__init__()

        self.net = nn.Sequential(Linear(input_dim,hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, input_dim]
        returns: tensor of shape [batch_size, num_actions] -- raw Q-values
        """
        return self.net(x)
        


class BanditPolicy:
    """
      - the optimizer (updates weights)
      - the loss function (measures how wrong a prediction was)
      - epsilon-greedy action selection
      - the update() method that performs one training step
    """

    def __init__(self, input_dim: int = 16, lr: float = 1e-3, epsilon: float = 0.15):

        self.network = QNetwork(input_dim=input_dim, num_actions=len(ACTIONS))

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.loss_fn = nn.MSELoss()

        self.epsilon = epsilon  # probability of picking a random action instead of the network's best guess

    # ------------------------------------------------------------------
    # ACTION SELECTION -- "the guess"
    # ------------------------------------------------------------------
    def select_action(self, state_vector: List[float], explore: bool = True) -> Tuple[str, torch.Tensor]:
      
        sample = torch.Tensor(state_vector).unsqueeze(0)
        q_values = self.network(sample).squeeze(0)

        idx = None

        if explore and random.random() < self.epsilon:
            idx = random.choice(range(len(ACTIONS)))
        else:
            idx = torch.argmax(q_values).item()
        
        return ACTIONS[idx], q_values


    # ------------------------------------------------------------------
    # LEARNING STEP -- "the correction"
    # ------------------------------------------------------------------
    def update(self, state_vector: List[float], action: str, observed_reward: float) -> float:
        """
        Called AFTER the simulator has actually applied `action` and measured
        the real resulting reward. This is the only method that changes the
        network's weights.


        """
        sample = torch.Tensor(state_vector).unsqueeze(0)
        q_values = self.network(sample).squeeze(0)
        action_idx = ACTION_TO_IDX[action]
        predicted_q = q_values[action_idx]
        target_tensor = torch.tensor(observed_reward, dtype=torch.float32)
        loss = self.loss_fn(predicted_q, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save(self.network.state_dict(), path)
 
    def load(self, path: str):
        self.network.load_state_dict(torch.load(path))
 
 
if __name__ == "__main__":
 
 
    policy = BanditPolicy(input_dim=16, lr=0.01, epsilon=0.2)
 
    def fake_environment(state_vector, action):
        cardinality = state_vector[8]
        if action == "INDEX":
            return 1.0 if cardinality > 0.5 else -1.0
        return 0.0
 
    losses = []
    NUM_STEPS = 2000
    for step in range(NUM_STEPS):
        fake_state = [0.0] * 16
        fake_state[8] = random.random()  # random cardinality each step
 
        action, q_values = policy.select_action(fake_state, explore=True)
        reward = fake_environment(fake_state, action)
        loss = policy.update(fake_state, action, reward)
        losses.append(loss)
 
    print("Average loss, first 100 steps:", sum(losses[:100]) / 100)
    print("Average loss, last 100 steps: ", sum(losses[-100:]) / 100)
 
    high_card_state = [0.0] * 16
    high_card_state[8] = 0.9
    action, q_values = policy.select_action(high_card_state, explore=False)
    print(f"\nHigh cardinality (0.9) -> chosen action: {action}, Q-values: {q_values.tolist()}")
 
    low_card_state = [0.0] * 16
    low_card_state[8] = 0.1
    action, q_values = policy.select_action(low_card_state, explore=False)
    print(f"Low cardinality (0.1)  -> chosen action: {action}, Q-values: {q_values.tolist()}")
 