import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from agent import Agent
import custom_envs

import matplotlib

from envs import make_env_discrete, make_env_continuous
from simulate import simulate

matplotlib.use('Agg')

def simulate_all_agent_types(num_steps=10000, episodes_per_env=5, num_envs=4):
    """
    Create and simulate each type of agent for a given number of steps.
    Uses randomly initialized policies.

    Args:
        num_steps: Maximum number of steps to simulate per environment
        episodes_per_env: Number of episodes to run per environment
        num_envs: Number of environments to vectorize

    Returns:
        Dictionary with results for each environment
    """
    # Create test environments covering observation and action space combinations
    # Excluding discrete_obs_continuous_action as it's not used in practice
    env_configs = {
        'discrete_obs_discrete_action': {
            'env_id': 'GridWorld-5x5-v0',
            'make_env_fn': make_env_discrete,
            'kwargs': {}
        },
        'continuous_obs_discrete_action': {
            'env_id': 'CartPole-v1',
            'make_env_fn': make_env_discrete,
            'kwargs': {}
        },
        'continuous_obs_continuous_action': {
            'env_id': 'Pendulum-v1',
            'make_env_fn': make_env_continuous,
            'kwargs': {'gamma': 0.99}
        }
    }

    results = {}

    # Simulate each environment type
    for env_name, config in env_configs.items():
        # Create vectorized environment
        envs = gym.vector.SyncVectorEnv(
            [config['make_env_fn'](
                config['env_id'],
                i,
                capture_video=False,
                run_name=f"videos/{env_name}",
                **config['kwargs']
            ) for i in range(num_envs)]
        )

        # Create agent for this environment
        agent = Agent(envs)

        # Simulate using the vectorized environment
        print(f"Simulating {env_name}...")
        return_avg, return_std, success_avg, success_std = simulate(
            envs, agent, episodes_per_env, num_steps
        )

        # Record results
        results[env_name] = {
            'return_avg': return_avg,
            'return_std': return_std,
            'success_avg': success_avg,
            'success_std': success_std
        }

        print(f"  Average return: {return_avg:.2f} ± {return_std:.2f}")
        print(f"  Success rate: {success_avg:.2f} ± {success_std:.2f}")

        # Close environment
        envs.close()

    return results


if __name__ == "__main__":
    # Run simulation
    results = simulate_all_agent_types(num_steps=np.inf, episodes_per_env=100, num_envs=1)

    # Plot results
    env_names = list(results.keys())
    returns = [results[env]['return_avg'] for env in env_names]
    return_errors = [results[env]['return_std'] for env in env_names]

    plt.figure(figsize=(10, 6))
    plt.bar(env_names, returns, yerr=return_errors, capsize=10)
    plt.ylabel('Average Return')
    plt.title('Performance of Different Agent Types')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('agent_performance.png')
    plt.show()