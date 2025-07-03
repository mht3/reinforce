import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
import wandb
from torch.utils.data import TensorDataset, DataLoader
from .buffers import RolloutBuffer

class REINFORCE:
    '''
    Custom implementation of the REINFORCE algorithm (monte carlo policy gradient).
    REINFORCE has a high variance in comparison to more modern algorithms like PPO.

    Args:
        policy (torch.nn.Module): neural network initialization for policy network
        env (gymnasium.Env): Gym environment.
        verbose (bool): flag for debugging.
        optimizer (torch.optim.Optimizer): optimizer for the policy network. Defaults to Adam.
        seed (int): random seed for initialization
        wandb_name (str): weights and biases experiment name. Defaults to None (no wandb)
        baseline (bool): use value baseline
        shared_value_network (bool): share feature extractor for policy and value net
        vf_coef (float): coefficient for value loss in shared network loss (default 0.5)
    ''' 
    def __init__(self, policy:torch.nn.Module, env:gym.Env, verbose:bool=False,
                 gamma:float = 0.99, seed:int=None, wandb_name:str=None,
                 baseline:bool=True, shared_value_network:bool=False, vf_coef:float=0.5):
        self.env = env
        self.action_space = env.action_space
        if isinstance(self.action_space, spaces.Discrete):
            self.actions_are_continuous = False
        elif isinstance(self.action_space, spaces.Box):
            self.actions_are_continuous = True
         
        self.policy = policy
        
        self.verbose = verbose
        self.gamma = gamma
        if seed is not None:
            self.seed = seed
            REINFORCE.set_seed(seed)
        self.rollout_buffer = RolloutBuffer()
        self.wandb_name = wandb_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.baseline = baseline
        self.shared_value_network = shared_value_network
        self.vf_coef = vf_coef

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _evaluate_actions(self, obs, action):
        policy_output = self.policy(obs)
        
        if isinstance(policy_output, tuple):
            # reinforce with baseline
            policy_logits, values = policy_output
        else:
            policy_logits = policy_output
        
        if self.actions_are_continuous:
            mean = policy_logits
            std = torch.exp(self.policy.log_std)
            distribution = torch.distributions.Normal(mean, std)
            log_prob = distribution.log_prob(action)
        else:
            # discrete actions
            probs = torch.nn.functional.softmax(policy_logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)
            log_prob = distribution.log_prob(action)

        if len(log_prob.shape) > 1:
            log_prob = log_prob.sum(dim=-1)

        if self.baseline:
            return log_prob, values.flatten()
        else:
            return log_prob, None

    def _get_action(self, obs, deterministic=False):
        policy_output = self.policy(obs)
        
        if isinstance(policy_output, tuple):
            # reinforce with baseline
            policy_logits, _ = policy_output
        else:
            policy_logits = policy_output

        if self.actions_are_continuous:
            mean = policy_logits
            std = torch.exp(self.policy.log_std)
            distribution = torch.distributions.Normal(mean, std)
            action = distribution.mean if deterministic else distribution.sample()
        else:
            # discrete actions need softmax to get the probabilities
            probs = torch.nn.functional.softmax(policy_logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)
            action = torch.argmax(distribution.probs, dim=1) if deterministic else distribution.sample()
        return action 

    def _collect_rollouts(self):
        '''
        Collect state, action, reward pairs for a single episode.
        '''
        obs, info = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        truncate = False
        terminate = False
        while not truncate and not terminate:
            action = self._get_action(obs)
            if self.actions_are_continuous:
                action_clipped = np.clip(action.detach().numpy(), self.action_space.low, self.action_space.high)
            else:
                action_clipped = action.detach().numpy()

            next_obs, reward, terminate, truncate, info = self.env.step(action_clipped)
            self.rollout_buffer.add(obs, action, reward)
            obs = torch.tensor(next_obs, dtype=torch.float32)
            self.total_steps += 1

        # get cost to go from rewards
        self.rollout_buffer.returns = self._compute_cost_to_go(self.rollout_buffer.rewards, gamma=self.gamma)
        
    def _compute_cost_to_go(self, rewards, gamma):
        '''
        Compute the cost to go, G, for each timestep.
        Returns:
            Gs (list): cost-to-go for each timestep.
        '''
        Gs = []
        for t in range(len(rewards)):
            G = 0.
            for k, rew in enumerate(rewards[t:]):
                G += gamma**k * rew
            Gs.append(G)
        return Gs

    def _update_policy(self):
        self.optimizer.zero_grad()
        if self.baseline and not self.shared_value_network:
            self.value_optimizer.zero_grad()

        obs_tensor = torch.stack(self.rollout_buffer.states).to(self.device)
        actions_tensor = torch.stack(self.rollout_buffer.actions).to(self.device)
        cost_to_go_tensor = torch.tensor(self.rollout_buffer.returns, dtype=torch.float32).to(self.device)
        # set batch_size to dataset length if batch_size = -1
        batch_size = len(obs_tensor) if self.batch_size == -1 else self.batch_size
        dataset = TensorDataset(obs_tensor, actions_tensor, cost_to_go_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            batch_obs, batch_actions, batch_cost_to_go = batch
            log_probs, values = self._evaluate_actions(batch_obs, batch_actions)
            if self.baseline:
                # REINFORCE with baseline: delta = G - b(S)
                delta = (batch_cost_to_go - values).detach()
                # normalize
                # if batch_obs.shape[0] > 1:
                #     delta = (delta - delta.mean()) / (delta.std() + 1e-6)
                policy_loss = torch.mean(-log_probs * delta)
                value_loss = torch.mean(torch.square(values - batch_cost_to_go))
            else:
                # REINFORCE
                # multiply by gamma^t per sutton and barto (separate from cost to go)
                policy_loss = torch.mean(-log_probs * batch_cost_to_go)

            if self.shared_value_network and self.baseline:
                total_loss = policy_loss + self.vf_coef * value_loss
                total_loss.backward()
            else:
                policy_loss.backward()
                # value net
                if self.baseline:
                    value_loss.backward()
            # losses
            total_policy_loss += policy_loss.item() * batch_obs.shape[0]
            total_samples += batch_obs.shape[0]
            avg_value_loss = 0.0
            if self.baseline:
                total_value_loss += value_loss.item() * batch_obs.shape[0]

        self.optimizer.step()
        avg_policy_loss = total_policy_loss / total_samples
        if self.baseline:
            avg_value_loss = total_value_loss / total_samples
            if not self.shared_value_network:
                self.value_optimizer.step()

        return avg_policy_loss, avg_value_loss

    def learn(self, num_episodes:int, batch_size=128, lr:float=3e-4,
              policy_optimizer:torch.optim.Optimizer=None, value_optimizer:torch.optim.Optimizer=None, num_steps:int=1000000):
        '''
        Outer training loop for REINFORCE algorithm. Initialias the optimizer and trains for a total number of episodes.

        Args:
            num_episodes (int): total number of episodes for training
            lr (float): learning rate
            optimizer (torch.optim.Optimizer): optimizer for the policy network. Defaults to Adam.
            num_steps (int): maximum total environment steps
        ''' 
        self.optimizer = policy_optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        if self.shared_value_network:
            self.value_optimizer = self.optimizer
        else:
            self.value_optimizer = value_optimizer
            if self.value_optimizer is None:
                self.value_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.batch_size=batch_size
        self.max_steps = num_steps
        # wandb setup
        if self.wandb_name is not None:
            config = {"lr": lr,
                      "num_episodes": num_episodes,
                      "batch_size": batch_size,
                      "env": self.env,
                      "optimizer": self.optimizer,
                      "gamma": self.gamma,
                      "seed": self.seed,
                      "baseline": self.baseline,
                      "actions_are_continuous": self.actions_are_continuous,
                      "shared_value_network": self.shared_value_network,
                      "max_steps": self.max_steps,
                      }
            wandb.init(
                project="Reinforce",
                name=self.wandb_name,
                config=config
            )

        self.train(num_episodes)

        if self.wandb_name is not None:
            wandb.finish()

    def train(self, num_episodes):
        '''
        Updates policy using rollout buffer.
        '''
        self.policy.train()
        # initialize total steps to 0
        self.total_steps = 0
        use_steps_bar = num_episodes is None
        if use_steps_bar:
            num_episodes = 1e9
            pbar = tqdm(total=self.max_steps, desc="Train", unit="steps")
        else:
            pbar = tqdm(range(num_episodes), desc="Train", unit="episodes")
        episode = 0
        while episode < num_episodes and self.total_steps < self.max_steps:
            steps_before = self.total_steps
            self.rollout_buffer.reset()
            with torch.no_grad():
                # no gradient tracking when collecting rollouts
                self._collect_rollouts()
            # update policy network
            policy_loss, value_loss = self._update_policy()
            ep_reward = np.sum(self.rollout_buffer.rewards)
            steps_this_episode = self.total_steps - steps_before
            if use_steps_bar:
                pbar.update(steps_this_episode)
            else:
                pbar.update(1)
            pbar.set_postfix({"ep_reward": "{:.4f}".format(ep_reward)})
            if self.wandb_name is not None:
                self.log_metrics(ep_reward, policy_loss, value_loss)
            episode += 1
        pbar.close()

    def log_metrics(self, ep_reward, policy_loss, value_loss=0.):
        ep_length = len(self.rollout_buffer.rewards)
        metrics = {'rollout/ep_rew' : ep_reward, 'rollout/ep_len' : ep_length, 'global_step': self.total_steps, 
                   'train/policy_gradient_loss' : policy_loss, 'train/learning_rate':  self.optimizer.param_groups[0]['lr']}
        
        if self.baseline:
            metrics['train/value_loss'] = value_loss
            metrics['train/loss'] = policy_loss + self.vf_coef * value_loss
            if not self.shared_value_network:
                metrics['train/value_learning_rate'] =  self.value_optimizer.param_groups[0]['lr']
        if self.actions_are_continuous:
            metrics['train/std'] = torch.exp(self.policy.log_std).mean()

        wandb.log(metrics)

    def predict(self, obs, deterministic=False):
        self.policy.eval()
        action = self._get_action(obs, deterministic)
        return action
    
    def save(self, filename):
        '''
        save the policy network weights
        '''
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        '''
        load the policy network from a .zip or .pt file
        '''
        self.policy.load_state_dict(torch.load(filename, weights_only=True))
        return self