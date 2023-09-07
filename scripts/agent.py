import numpy as np
import random
import copy
from scripts.model import Actor, Critic
from sklearn.base import BaseEstimator
import torch
import torch.nn.functional as F
import torch.optim as optim
from scripts.PriorityReplayBuffer import PriorityReplayBuffer
from scripts.ReplayBuffer import ReplayBuffer
from torch.distributions import Categorical


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3             # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 8e-4        # learning rate of the critic
WEIGHT_DECAY = 0.00001   # L2 weight decay
TIME = 64       # Train network every TIME time step
TRAIN_NUM = 128  # Train network TRAIN_NUM times 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(BaseEstimator):
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size,num_agents, random_seed ):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        # Critic Network (w/ Target Network)
        self.actor_local  = Actor(state_size, action_size, 256).to(device) 

        self.actor_target = Actor(state_size, action_size, 256).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) 

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, 256).to(device) 
        
        self.critic_target = Critic(state_size, action_size, 256).to(device) 
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) 
        
        # Noise process
        self.noise = OUNoise(self.action_size, mu=0.0, theta=0.1, sigma=0.3)
        
        # Replay memory
#         self.pbuffer = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.pbuffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,self.seed)

        self.deep_copy(self.actor_target, self.actor_local)
        self.deep_copy(self.critic_target, self.critic_local)
        self.num_agents = num_agents
        
    def step(self, states, actions, rewards, next_states, dones, time_stamp):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
#         priority = 1.0
        for i in range(self.num_agents):
            self.pbuffer.add(states[i,:], actions[i], rewards[i], next_states[i,:], dones[i])

        t = time_stamp%TIME
        # Learn, if enough samples are available in memory
        seen = []
        if t == 0 and len(self.pbuffer) >= BATCH_SIZE:
            for _ in range(TRAIN_NUM):
                p_sample = self.pbuffer.sample()
                if p_sample[0].mean() not in seen: # encourage unseen sample in one episode
                    seen.append(p_sample[0].mean())
                    self.learn(p_sample, GAMMA)
                    
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i,:]).unsqueeze(0)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).squeeze(0).detach().numpy()
                action = np.clip(action, -1, 1)
            self.actor_local.train()
            if add_noise:
                noise = self.noise.sample()/5
                action += noise
            actions.append(action)
            
        return actions

    def reset(self):
        self.noise.reset()

    def learn(self, p_sample, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
#         batch_indices, batch_experiences, batch_weights  = p_sample
#         states, actions, rewards, next_states, dones = zip(*batch_experiences)
#         states = torch.FloatTensor(states)
#         actions = torch.FloatTensor(actions)
#         rewards = torch.FloatTensor(rewards).unsqueeze(1)
#         next_states = torch.FloatTensor(next_states)
#         dones = torch.FloatTensor(dones).unsqueeze(1)
#         weights = torch.FloatTensor(batch_weights).unsqueeze(1)

        states, actions, rewards, next_states, dones = p_sample

        # ---------------------------- update critic ---------------------------- #
        # Input State to the actor network and get the best believed action to take 
        actions_next = self.actor_target(next_states)
        ## CRITIC
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        # Compute critic loss between expected and target
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.5)  # Apply gradient clipping
        self.critic_optimizer.step()

        ## ACTOR
        actions_pred = self.actor_local(states)
        actor_loss   = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the losses
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.5)
        self.actor_optimizer.step()
        
        
        # ----------------------- blending weights from local to target ------------#
#         errors = (Q_targets - Q_expected).detach().numpy()
#         self.pbuffer.update_priorities(batch_indices, np.abs(errors))
        self.soft_update(self.actor_local, self.actor_target, TAU) 
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def deep_copy(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class OUNoise:
    def __init__(self, size, mu, theta, sigma):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state   