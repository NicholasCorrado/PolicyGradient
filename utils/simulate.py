from collections import defaultdict

import numpy as np
import torch

def simulate(env, actor, eval_episodes, eval_steps=np.inf):
    logs = defaultdict(list)
    step = 0
    for episode_i in range(eval_episodes):
        logs_episode = defaultdict(list)

        obs, _ = env.reset()
        done = False

        while not done:

            with torch.no_grad():
                actions = actor.get_action(torch.Tensor(obs).to('cpu'), sample=False)
                actions = actions.cpu().numpy()

            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            done = np.logical_or(terminateds, truncateds)

            obs = next_obs
            logs_episode['rewards'].append(rewards)

            step += 1

            if step >= eval_steps:
                break
        if step >= eval_steps:
            break

        logs['returns'].append(np.sum(logs_episode['rewards']))
        try:
            logs['successes'].append(infos['is_success'])
        except:
            logs['successes'].append(False)

    return_avg = np.mean(logs['returns'])
    return_std = np.std(logs['returns'])
    success_avg = np.mean(logs['successes'])
    success_std = np.std(logs['successes'])
    return return_avg, return_std, success_avg, success_std
