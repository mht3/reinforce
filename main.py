import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from src.models import PolicyNN
from src import REINFORCE

def run_single_episode(model, env):
    terminate = False
    truncate = False
    obs, info = env.reset()
    rew = 0.
    while not truncate and not terminate:
        obs = torch.tensor(obs, dtype=torch.float32)
        action = model.predict(obs).detach().numpy()
        obs, reward, terminate, truncate, info = env.step(action)
        rew += reward
    return rew

def test_model(policy, env, num_episodes=1):
    total_reward = 0.
    for i in range(num_episodes):
        total_reward += run_single_episode(policy, env)
    avg_reward = total_reward / num_episodes
    print("Average Reward: {}".format(avg_reward))

def train_model(env, lr, num_episodes, batch_size, baseline, shared_value_network, seed, verbose, name, save_path, num_steps, vf_coef):
    n_input = env.observation_space.shape[0]
    if isinstance(env.action_space, spaces.Box):
        n_output = env.action_space.shape[0]
        continuous_actions = True
    else:
        n_output = env.action_space.n
        continuous_actions = False

    policy = PolicyNN(n_input, n_output, continuous_actions=continuous_actions, value_network=baseline,
                      shared_value_network=shared_value_network)

    model = REINFORCE(policy, env, verbose=verbose, seed=seed, 
                      baseline=baseline, shared_value_network=shared_value_network, 
                      wandb_name=name, vf_coef=vf_coef)

    model.learn(num_episodes=num_episodes, lr=lr, batch_size=batch_size, num_steps=num_steps)
    model.save(save_path)

    # test trained model
    test_env = gym.make(args.env_name, render_mode=None)
    trained_model = model.load(save_path)
    test_model(trained_model, test_env, num_episodes=10)

    return model

def main(args):
    # make gym environment
    env = gym.make(args.env_name)
    # rwb for reinforce with baseline or r for reinforce
    baseline_str = "rwb" if args.baseline else "r"
    if args.shared_value_network:
        baseline_str += '_s'
    name = f"{args.env_name}_{baseline_str}_lr{args.lr}_bs{args.batch_size}_s{args.seed}"
    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(cur_path, 'saved_models', name + '.zip')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    # train model
    model = train_model(env, lr=args.lr, num_episodes=args.num_episodes, batch_size=args.batch_size, baseline=args.baseline,
                        shared_value_network=args.shared_value_network, seed=args.seed,
                        verbose=args.verbose, name=name, save_path=save_path, num_steps=args.num_steps, vf_coef=args.vf_coef)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='REINFORCE Algorithm')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_episodes', type=int, default=None, help='Number of training episodes (if not set, will be determined by num_steps)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. Defaults to 1.')
    parser.add_argument('--baseline', action='store_true', help='REINFORCE with baseline')
    parser.add_argument('--no_shared_value_network', dest='shared_value_network', action='store_false', help='Do NOT share feature extractor for policy and value net (default: shared)')
    parser.set_defaults(shared_value_network=True)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--num_steps', type=int, default=5000000, help='Maximum total environment steps to run (across all episodes).')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value loss coefficient for shared network loss (default 0.5)')
    
    args = parser.parse_args()

    main(args)