import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import numpy as np

def make_env_continuous(env_id, idx, capture_video=False, run_name="run", gamma=0.99):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def make_env_discrete(env_id, idx, capture_video=False, run_name="run"):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def make_env(env_id, idx, capture_video=False, run_name="run", gamma=0.99):
    # Create a temporary environment to check action space
    temp_env = gym.make(env_id)
    is_discrete_action = isinstance(temp_env.action_space, gym.spaces.Discrete)
    temp_env.close()

    # Choose the appropriate environment factory
    if is_discrete_action:
        return make_env_discrete(env_id, idx, capture_video, run_name)
    else:
        return make_env_continuous(env_id, idx, capture_video, run_name, gamma)