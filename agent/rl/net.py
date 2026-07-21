from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from .approximator import MLP


def load_bc_weights_partial(bc_model, rl_model, bc_feature_indices):
    """
    Transfer BC weights to RL model with expanded observation space.
    
    Args:
        bc_model: trained BCPolicy with obs_dim=2
        rl_model: new BCPolicy with obs_dim=6 (or more)
        bc_feature_indices: which indices in RL obs correspond to BC features
                           e.g., [0, 1] if headway_ratio and epsilon_arrival 
                           are the first two features in RL obs
    """
    bc_state = bc_model.state_dict()
    rl_state = rl_model.state_dict()
    
    for name, param in bc_state.items():
        if name == 'feature_net.0.weight':  # First linear layer
            # Shape: [hidden_dim, obs_dim]
            # Copy BC weights to corresponding feature columns
            for new_idx, old_idx in enumerate(bc_feature_indices):
                rl_state[name][:, old_idx] = param[:, new_idx]
            
            # Initialize NEW feature weights to near-zero
            all_indices = set(range(rl_state[name].shape[1]))
            new_indices = all_indices - set(bc_feature_indices)
            for idx in new_indices:
                nn.init.normal_(rl_state[name][:, idx], mean=0, std=0.01)
                
        elif name == 'feature_net.0.bias':  # First layer bias
            rl_state[name] = param.clone()
            
        else:  # All other layers — direct copy
            rl_state[name] = param.clone()
    
    rl_model.load_state_dict(rl_state)
    print(f"Transferred BC weights. New features initialized near-zero.")
    return rl_model


class BCPolicy(nn.Module):
    """
    Behavior Cloning Policy Network
    - Compatible with Stable Baselines3 / custom DDPG
    - Can be used as actor network initialization
    """
    def __init__(self, state_size: int, hidde_size: Tuple, init_type: str = 'default'):
        super(BCPolicy, self).__init__()
        
        
        # Build network
        layers = []
        prev_dim = state_size
        for hidden_dim in hidde_size:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Helps with RL stability
            ])
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights (important for RL)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for action head
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
    
    def forward(self, obs):
        features = self.feature_net(obs)
        action = self.action_head(features)
        # Bound action to valid range using sigmoid
        # action = torch.sigmoid(action) * (self.action_high - self.action_low) + self.action_low
        action = torch.sigmoid(action)  # Assuming action space is symmetric
        return action


class Actor_Net(torch.nn.Module):
    def __init__(self, state_size: int, hidde_size: Tuple, init_type: str = 'default'):
        ''' Actor network with 

        Args:
            state_size: size of the state (backward and forward spacing, and discrete state if any)
            hidde_size: tuple with the number of neurons in each hidden layer for continuous feature
        '''
        super(Actor_Net, self).__init__()
        self._mlp = MLP(state_size, 1, hidde_size,
                        outpu='sigmoid', init_type=init_type)

    def forward(self, x):
        # if type(x) == tuple:
        # x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        return self._mlp(x)


class Critic_Net(torch.nn.Module):
    def __init__(self, state_size, hidde_size, init_type='default'):
        self._state_size = state_size
        super(Critic_Net, self).__init__()
        self._mlp = MLP(self._state_size+1, 1, hidde_size,
                        outpu='logits', init_type=init_type)

    def forward(self, x):
        s = x[:, 0:self._state_size]
        a = x[:, self._state_size].unsqueeze(1)
        conti_x_a = torch.cat((s, a), dim=1)
        logit = self._mlp(conti_x_a)
        return logit
