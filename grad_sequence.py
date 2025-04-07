import glob
from collections import defaultdict

import matplotlib
import tyro
from stable_baselines3.common.utils import get_latest_run_id

from third_order_pg import compute_third_order_policy_gradient

from first_order_pg import compute_first_order_policy_gradient
from utils.agent import Agent
from utils.agent import ContinuousObsDiscreteActionAgent, ContinuousObsContinuousActionAgent
from wrappers import load_norm_stats

matplotlib.use('TkAgg')

# !/usr/bin/env python3
import argparse
import copy
import os
import gymnasium as gym
import custom_envs
import numpy as np
import torch
from tqdm import tqdm
from utils.envs import make_env

from dataclasses import dataclass, field
from typing import List, Optional

from utils.agent import Agent

@dataclass
class Args:
    env_id: str = "GridWorld-5x5-v0"
    n_samples_true: int = int(1e5)
    n_samples: int = int(10e3)
    run_id: int = None
    seed: int = 42
    cuda: bool = False
    overwrite_true_grad: bool = False
    output_rootdir: str = 'grad_results_tmp2'
    output_subdir: str = ''

def collect_transitions(agent, envs, num_transitions, random=True, device='cpu'):
    """Collect transitions using a fixed policy"""
    obs_buffer = torch.zeros((num_transitions, 1) + envs.single_observation_space.shape).to(device)
    action_buffer = torch.zeros((num_transitions, 1) + envs.single_action_space.shape).to(device)
    reward_buffer = torch.zeros((num_transitions, 1)).to(device)
    done_buffer = torch.zeros((num_transitions, 1)).to(device)
    terminated_buffer = torch.zeros((num_transitions, 1)).to(device)

    # Reset the environment at the beginning of collection
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)

    for step in tqdm(range(num_transitions), mininterval=1):
        obs_buffer[step] = next_obs
        done_buffer[step] = next_done

        with torch.no_grad():
            action = agent.get_action(next_obs)
        action_buffer[step] = action

        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)
        terminated_buffer[step] = torch.tensor(terminations).to(device)
        reward_buffer[step] = torch.tensor(reward).to(device).view(-1)

        next_obs = torch.Tensor(next_obs).to(device)

    return obs_buffer, action_buffer, reward_buffer, done_buffer


def compute_returns(rewards, dones, gamma=0.99):
    """Compute discounted returns more accurately by handling episode boundaries"""
    returns = torch.zeros_like(rewards)

    # Process each episode separately
    episode_ends = torch.where(dones.view(-1) == 1)[0].cpu().numpy()

    # Add the last index as an episode end if the last transition isn't already marked as done
    if len(episode_ends) == 0 or episode_ends[-1] != len(rewards) - 1:
        episode_ends = np.append(episode_ends, len(rewards) - 1)

    # Add -1 as the start of the first episode
    episode_starts = np.append([-1], episode_ends[:-1])

    # Process each episode
    for start, end in zip(episode_starts, episode_ends):
        running_return = 0.0
        for t in reversed(range(start + 1, end + 1)):
            running_return = rewards[t] + gamma * running_return * (1 - dones[t])
            returns[t] = running_return

    print(returns)
    return returns

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors"""
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)


def main():
    args = tyro.cli(Args)


    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, False, "gradient_eval")])

    policy_dirpath = f'results/{args.env_id}/ppo/on_policy/run_1'

    # for policy_path in sorted(glob.glob(os.path.join(policy_dirpath, "policy_*.pt"))):
    for policy_i in range(11):
        policy_path = f'{policy_dirpath}/policy_{policy_i}.pt'
        env_path = f'{policy_dirpath}/envs_{policy_i}.pt'
        output_dir = f'{args.output_rootdir}/{args.env_id}/{args.output_subdir}/policy_{policy_i}'
        if args.run_id:
            args.seed = args.run_id
            output_dir += f"/run_{args.run_id}/"
        else:
            args.seed = np.random.randint(2 ** 32 - 1)
            run_id = get_latest_run_id(log_path=output_dir, log_name='run') + 1
            output_dir += f"/run_{run_id}"
        os.makedirs(output_dir, exist_ok=True)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        print(policy_path)
        agent = Agent(envs)
        agent.load_state_dict(torch.load(policy_path, map_location=device, weights_only=False))
        load_norm_stats(envs, env_path)
        envs.envs[0].set_eval()
        agent.eval()  # Set to evaluation mode

        # Check if the true gradient has already been computed
        true_grad_path = f'{policy_dirpath}/true_grad_{policy_i}.pt'
        if os.path.exists(true_grad_path) and not args.overwrite_true_grad:
            print(f"Loading existing true gradient from {true_grad_path}")
            gradient_data = torch.load(true_grad_path, map_location=device)
            print('gradient data:', gradient_data)
            true_gradient_normalized = gradient_data['normalized']
            true_gradient_unnormalized = gradient_data['unnormalized']

        else:
            print(f"Computing 'true' policy gradient with {args.n_samples_true} samples...")
            obs_buffer, action_buffer, reward_buffer, done_buffer = collect_transitions(agent, envs, args.n_samples_true)
            returns = compute_returns(reward_buffer, done_buffer)
            true_gradient_unnormalized = compute_first_order_policy_gradient(agent, obs_buffer, action_buffer, returns)
            true_gradient_norm = torch.norm(true_gradient_unnormalized)
            true_gradient_normalized = true_gradient_unnormalized / true_gradient_norm

            # Save the true gradient
            os.makedirs(os.path.dirname(true_grad_path), exist_ok=True)
            torch.save({
                'normalized': true_gradient_normalized,
                'unnormalized': true_gradient_unnormalized,
                'norm': true_gradient_norm,
                'n_samples': args.n_samples_true,
            }, true_grad_path)

            print(f"True gradient saved to {true_grad_path}")

        # Results dictionary
        results = defaultdict(list)

        # Create a fresh copy of the policy for each experiment
        agent = copy.deepcopy(agent)

        # Set different seed for each experiment
        exp_seed = args.seed
        torch.manual_seed(exp_seed)
        np.random.seed(exp_seed)
        envs.reset(seed=exp_seed)

        obs, actions, rewards, dones = collect_transitions(agent, envs, args.n_samples, device)
        returns = compute_returns(rewards, dones)

        sample_sizes = np.logspace(np.log10(1024), np.log10(args.n_samples), 20).astype(int)
        for n in tqdm(sample_sizes, mininterval=1):
            empirical_gradient_unnormalized = compute_first_order_policy_gradient(agent, obs[:n], actions[:n], returns[:n])

            # Compute metrics
            empirical_gradient_norm = torch.norm(empirical_gradient_unnormalized)
            empirical_gradient_normalized = empirical_gradient_unnormalized / empirical_gradient_norm
            cos_sim = cosine_similarity(empirical_gradient_normalized, true_gradient_normalized)

            l2_distance_unnormalized = torch.norm(empirical_gradient_unnormalized - true_gradient_unnormalized)
            l2_distance_normalized = torch.norm(empirical_gradient_normalized - true_gradient_normalized)


            results['cosine_similarity'].append(cos_sim)
            results['norm'].append(empirical_gradient_norm)
            results['l2_distance_normalized'].append(l2_distance_normalized)
            results['l2_distance_unnormalized'].append(l2_distance_unnormalized)

            # for key, val in results:

        results['sample_size'] = sample_sizes
        np.savez(
            f'{output_dir}/evaluations.npz',
            **results,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"Error occurred: {e}")
        print(traceback.format_exc())